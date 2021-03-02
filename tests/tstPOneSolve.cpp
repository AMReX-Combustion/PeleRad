#include <boost/test/unit_test.hpp>

#define BOOST_TEST_MODULE ponesolveeb

#include <AMRParam.hpp>
#include <MLMGParam.hpp>
#include <POneEquation.hpp>

void init_prob(const Vector<Geometry>& geom, Vector<MultiFab>& alpha,
    Vector<MultiFab>& beta, Vector<MultiFab>& rhs, Vector<MultiFab>& exact,
    double const L)
{
    const int nlevels = geom.size();

    const double n        = 3.0;
    const double npioverL = n * M_PI / L;

    for (int ilev = 0; ilev < nlevels; ++ilev)
    {
        const double* problo = geom[ilev].ProbLo();
        const double* probhi = geom[ilev].ProbHi();
        const double* dx     = geom[ilev].CellSize();

        for (MFIter mfi(alpha[ilev]); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.validbox();

            double xc = (probhi[0] + problo[0]) / 2.0;
            double yc = (probhi[1] + problo[1]) / 2.0;
            double zc = (probhi[2] + problo[2]) / 2.0;

            const Dim3 lo = amrex::lbound(bx);
            const Dim3 hi = amrex::ubound(bx);

            auto const& alpha_ = alpha[ilev][mfi].array();
            auto const& beta_  = beta[ilev][mfi].array();
            auto const& exact_ = exact[ilev][mfi].array();
            auto const& rhs_   = rhs[ilev][mfi].array();

            for (int k = lo.z; k <= hi.z; ++k)
            {
                double z = problo[2] + dx[2] * ((double)k + 0.5);
                for (int j = lo.y; j <= hi.y; ++j)
                {
                    double y = problo[1] + dx[1] * ((double)j + 0.5);
                    for (int i = lo.x; i <= hi.x; ++i)
                    {
                        double x = problo[0] + dx[0] * ((double)i + 0.5);

                        alpha_(i, j, k) = 1.0;

                        beta_(i, j, k) = 1.0;

                        double sincossin = std::sin(npioverL * x)
                                           * std::cos(npioverL * y)
                                           * std::sin(npioverL * z);

                        exact_(i, j, k) = sincossin;

                        rhs_(i, j, k) = (beta_(i, j, k) * npioverL * npioverL
                                            + beta_(i, j, k))
                                        * sincossin;
                    }
                }
            }
        }
    }
};

std::vector<double> check_norm(
    Vector<MultiFab>& phi, Vector<MultiFab>& exact, int const nlevels)
{
    std::vector<double> eps;

    for (int ilev = 0; ilev < nlevels; ++ilev)
    {
        MultiFab mf(phi[ilev].boxArray(), phi[ilev].DistributionMap(), 1, 0);
        MultiFab::Copy(mf, phi[ilev], 0, 0, 1, 0);

        MultiFab::Subtract(mf, exact[ilev], 0, 0, 1, 0);

        double L0norm = mf.norm0();
        double L1norm = mf.norm1();
        std::cout << "Level=" << ilev << ", L0 norm:" << L0norm
                  << ", L1 norm:" << L1norm << std::endl;
        eps.push_back(L1norm);
    }
    return eps;
}

BOOST_AUTO_TEST_CASE(p1_solve)
{
    amrex::ParmParse pp;
    PeleRad::AMRParam amrpp(pp);
    PeleRad::MLMGParam mlmgpp(pp);

    // initialize the grid
    const int nlevels       = amrpp.max_level_ + 1;
    const int n_cell        = amrpp.n_cell_;
    const int max_grid_size = amrpp.max_grid_size_;
    const int ref_ratio     = amrpp.ref_ratio_;

    std::cout << "n_cell=" << n_cell << ", nlevels=" << nlevels
              << ", max_grid_siz=" << max_grid_size
              << ", ref_ratio=" << ref_ratio << "\n";

    Vector<Geometry> geom(nlevels);
    Vector<BoxArray> grids(nlevels);
    Vector<DistributionMapping> dmap(nlevels);

    Box domain0(IntVect { AMREX_D_DECL(0, 0, 0) },
        IntVect { AMREX_D_DECL(n_cell - 1, n_cell - 1, n_cell - 1) });
    BoxArray ba0 { domain0 };

    grids[0] = ba0;
    grids[0].maxSize(max_grid_size);

    for (int ilev = 1; ilev < grids.size(); ++ilev)
    {
        ba0.grow(-n_cell / 4);
        ba0.refine(ref_ratio);
        grids[ilev] = ba0;
        grids[ilev].maxSize(max_grid_size);
    }

    for (int ilev = 0; ilev < nlevels; ++ilev)
    {
        dmap[ilev].define(grids[ilev]);
    }

    const double L = 2.0;

    amrex::RealBox rb { AMREX_D_DECL(-1.0, -1.0, -1.0),
        AMREX_D_DECL(1.0, 1.0, 1.0) };

    const int coord = 0;

    std::array<int, AMREX_SPACEDIM> is_periodic { AMREX_D_DECL(0, 0, 0) };

    auto bc_type = MLLinOp::BCType::Dirichlet;

    if (bc_type == MLLinOp::BCType::Periodic)
    {
        std::fill(is_periodic.begin(), is_periodic.end(), 1);
    }

    geom[0].define(domain0, &rb, coord, is_periodic.data());

    for (int ilev = 1; ilev < grids.size(); ++ilev)
    {
        domain0.refine(amrpp.ref_ratio_);
        geom[ilev].define(domain0, &rb, coord, is_periodic.data());
    }

    Vector<MultiFab> soln(nlevels);
    Vector<MultiFab> exact(nlevels);
    Vector<MultiFab> alpha(nlevels);
    Vector<MultiFab> beta(nlevels);
    Vector<MultiFab> rhs(nlevels);

    for (int ilev = 0; ilev < nlevels; ++ilev)
    {
        soln[ilev].define(grids[ilev], dmap[ilev], 1, 1);
        exact[ilev].define(grids[ilev], dmap[ilev], 1, 0);
        alpha[ilev].define(grids[ilev], dmap[ilev], 1, 0);
        beta[ilev].define(grids[ilev], dmap[ilev], 1, 1);
        rhs[ilev].define(grids[ilev], dmap[ilev], 1, 0);
    }

    PeleRad::POneEquation rte(amrpp, mlmgpp, geom, grids, dmap);

    init_prob(geom, alpha, beta, rhs, exact, L);

    for (auto& mf : soln)
    {
        mf.setVal(0.0);
    }

    rte.solve(soln, alpha, beta, rhs, exact);

    // turn off write for unit tests
    bool unittest = false;

    rte.write(soln, alpha, beta, rhs, exact, unittest);

    auto eps = check_norm(soln, exact, nlevels);

    for (auto iter : eps)
    {
        BOOST_TEST(iter < 1e-2 * n_cell * n_cell * n_cell);
    }
}
