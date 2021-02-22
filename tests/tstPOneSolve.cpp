#include <boost/test/unit_test.hpp>

#define BOOST_TEST_MODULE ponesolveeb

#include <AMRParam.hpp>
#include <MLMGParam.hpp>
#include <POneEquation.hpp>

void init_prob(const Vector<Geometry>& geom, Vector<MultiFab>& alpha,
    Vector<MultiFab>& beta, Vector<MultiFab>& rhs, Vector<MultiFab>& exact)
{
    Real a                  = 1e-3;
    Real b                  = 1.0 / 3.0;
    Real sigma              = 10.0;
    Real w                  = 0.05;
    MLLinOp::BCType bc_type = MLLinOp::BCType::Dirichlet;

    char bct;
    if (bc_type == MLLinOp::BCType::Dirichlet)
    {
        bct = 'd';
    }
    else if (bc_type == MLLinOp::BCType::Neumann)
    {
        bct = 'n';
    }
    else
    {
        bct = 'p';
    }

    const int nlevels = geom.size();

    const double tpi = 2.0 * M_PI;
    const double fpi = 4.0 * M_PI;
    const double fac = 12.0 * M_PI * M_PI;

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

            double theta = 0.5 * std::log(3.0) / (w + 1e-50);

            const Dim3 lo = amrex::lbound(bx);
            const Dim3 hi = amrex::ubound(bx);

            auto const& alpha_ = alpha[ilev][mfi].array();
            auto const& beta_  = beta[ilev][mfi].array();
            auto const& exact_ = exact[ilev][mfi].array();
            auto const& rhs_   = rhs[ilev][mfi].array();

            for (int k = lo.z - 1; k <= hi.z + 1; ++k)
            {
                // double z = problo[2] + dx[2] * ((double)k + 0.5);
                for (int j = lo.y - 1; j <= hi.y + 1; ++j)
                {
                    // double y = problo[1] + dx[1] * ((double)j + 0.5);
                    for (int i = lo.x - 1; i <= hi.x + 1; ++i)
                    {
                        /*
                        double x       = problo[0] + dx[0] * ((double)i + 0.5);
                        double r       = std::sqrt((x - xc) * (x - xc)
                                             + (y - yc) * (y - yc)
                                             + (z - zc) * (z - zc));
                        beta_(i, j, k) = (sigma - 1.0) / 2.0
                                             * std::tanh(theta * (r - 0.25))
                                         + (sigma + 1.0) / 2.0;*/
                        beta_(i, j, k) = 1.0;
                        // std::cout << beta_(i, j, k) << std::endl;
                    }
                }
            }

            for (int k = lo.z; k <= hi.z; ++k)
            {
                double z = problo[2] + dx[2] * ((double)k + 0.5);
                for (int j = lo.y; j <= hi.y; ++j)
                {
                    double y = problo[1] + dx[1] * ((double)j + 0.5);
                    for (int i = lo.x; i <= hi.x; ++i)
                    {
                        double x = problo[0] + dx[0] * ((double)i + 0.5);
                        double r = std::sqrt((x - xc) * (x - xc)
                                             + (y - yc) * (y - yc)
                                             + (z - zc) * (z - zc));
                        double coshtheta = std::cosh(theta * (r - 0.25));
                        double dbdrfac   = (sigma - 1.0) / 2.0
                                         / (coshtheta * coshtheta) * theta / r;
                        dbdrfac *= b;

                        alpha_(i, j, k) = 1.0;

                        double coscoscostpi = std::cos(tpi * x)
                                              * std::cos(tpi * y)
                                              * std::cos(tpi * z);
                        double coscoscosfpi = std::cos(fpi * x)
                                              * std::cos(fpi * y)
                                              * std::cos(fpi * z);

                        double sincoscos
                            = tpi * std::sin(tpi * x) * std::cos(tpi * y)
                                  * std::cos(tpi * z)
                              + M_PI * std::sin(fpi * x) * std::cos(fpi * y)
                                    * std::cos(fpi * z);

                        double cossincos
                            = tpi * std::cos(tpi * x) * std::sin(tpi * y)
                                  * std::cos(tpi * z)
                              + M_PI * std::cos(fpi * x) * std::sin(fpi * y)
                                    * std::cos(fpi * z);

                        double coscossin
                            = tpi * std::cos(tpi * x) * std::cos(tpi * y)
                                  * std::sin(tpi * z)
                              + M_PI * std::cos(fpi * x) * std::cos(fpi * y)
                                    * std::sin(fpi * z);

                        double sinsinsintpi = std::sin(tpi * x)
                                              * std::sin(tpi * y)
                                              * std::sin(tpi * z);
                        double sinsinsinfpi = std::sin(fpi * x)
                                              * std::sin(fpi * y)
                                              * std::sin(fpi * z);

                        double cossinsin
                            = -tpi * std::cos(tpi * x) * std::sin(tpi * y)
                                  * std::sin(tpi * z)
                              - M_PI * std::cos(fpi * x) * std::sin(fpi * y)
                                    * std::sin(fpi * z);

                        double sincossin
                            = -tpi * std::sin(tpi * x) * std::cos(tpi * y)
                                  * std::sin(tpi * z)
                              - M_PI * std::sin(fpi * x) * std::cos(fpi * y)
                                    * std::sin(fpi * z);

                        double sinsincos
                            = -tpi * std::sin(tpi * x) * std::sin(tpi * y)
                                  * std::cos(tpi * z)
                              - M_PI * std::sin(fpi * x) * std::sin(fpi * y)
                                    * std::cos(fpi * z);

                        if (bct == 'p' || bct == 'n')
                        {
                            exact_(i, j, k)
                                = 1.0 * coscoscostpi + 0.25 * coscoscosfpi;
                            rhs_(i, j, k)
                                = beta_(i, j, k) * b * fac
                                      * (coscoscostpi + coscoscosfpi)
                                  + dbdrfac
                                        * ((x - xc) * sincoscos
                                            + (y - yc) * cossincos
                                            + (z - zc) * coscossin)
                                  + a * (coscoscostpi + 0.25 * coscoscosfpi);
                        }
                        else
                        {
                            exact_(i, j, k)
                                = 1.0 * sinsinsintpi + 0.25 * sinsinsinfpi;
                            rhs_(i, j, k)
                                = beta_(i, j, k) * b * fac
                                      * (sinsinsintpi + sinsinsinfpi)
                                  + dbdrfac
                                        * ((x - xc) * cossinsin
                                            + (y - yc) * sincossin
                                            + (z - zc) * sinsincos)
                                  + a * (sinsinsintpi + 0.25 * sinsinsinfpi);
                            exact_(i, j, k) = 0.0;
                            rhs_(i, j, k)   = 1.0;
                        }
                    }
                }
            }
        }
    }
};

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

    Vector<amrex::Geometry> geom;
    Vector<BoxArray> grids;

    geom.resize(nlevels);
    grids.resize(nlevels);

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
        DistributionMapping dm { grids[ilev] };
        soln[ilev].define(grids[ilev], dm, 1, 1);
        exact[ilev].define(grids[ilev], dm, 1, 0);
        alpha[ilev].define(grids[ilev], dm, 1, 0);
        beta[ilev].define(grids[ilev], dm, 1, 1);
        rhs[ilev].define(grids[ilev], dm, 1, 0);
    }

    PeleRad::POneEquation rte(amrpp, mlmgpp, geom, grids);

    init_prob(geom, alpha, beta, rhs, exact);

    for (auto& mf : soln)
    {
        mf.setVal(0.0);
    }

    rte.solve(soln, alpha, beta, rhs, exact);

    //turn off write for unit tests
    bool unittest = true;

    rte.write(soln, alpha, beta, rhs, exact, unittest);

    BOOST_TEST(true);
}
