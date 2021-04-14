#include <boost/test/unit_test.hpp>

#define BOOST_TEST_MODULE ponerobin

#include <AMReX_PlotFileUtil.H>
#include <POneSingle.hpp>

AMREX_GPU_DEVICE AMREX_FORCE_INLINE void actual_init_abeclap(int i, int j,
    int k, amrex::Array4<amrex::Real> const& rhs,
    amrex::Array4<amrex::Real> const& sol,
    amrex::Array4<amrex::Real> const& alpha,
    amrex::Array4<amrex::Real> const& beta,
    amrex::Array4<amrex::Real> const& robin_a,
    amrex::Array4<amrex::Real> const& robin_b,
    amrex::Array4<amrex::Real> const& robin_f,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& prob_lo,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& prob_hi,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
    const amrex::Dim3& dlo, const amrex::Dim3& dhi, amrex::Box const& vbx,
    int robin_dir, int robin_face)
{
    double const L            = 2.0;
    double const n            = 3.0;
    double const npioverL     = n * M_PI / L;
    constexpr amrex::Real pi  = M_PI;
    constexpr amrex::Real tpi = 2.0 * pi;
    constexpr amrex::Real fpi = 4.0 * pi;

    amrex::Real xc = (prob_hi[0] + prob_lo[0]) * 0.5;
    amrex::Real yc = (prob_hi[1] + prob_lo[1]) * 0.5;
    amrex::Real zc = (prob_hi[2] + prob_lo[2]) * 0.5;

    amrex::Real x = prob_lo[0] + dx[0] * (i + 0.5);
    amrex::Real y = prob_lo[1] + dx[1] * (j + 0.5);
    amrex::Real z = prob_lo[2] + dx[2] * (k + 0.5);
    amrex::Real r = std::sqrt(
        (x - xc) * (x - xc) + (y - yc) * (y - yc) + (z - zc) * (z - zc));

    beta(i, j, k) = 1.0;

    x = amrex::min(amrex::max(x, prob_lo[0]), prob_hi[0]);
    y = amrex::min(amrex::max(y, prob_lo[1]), prob_hi[1]);
    z = amrex::min(amrex::max(z, prob_lo[2]), prob_hi[2]);

    double sincossin = std::sin(npioverL * x) * std::cos(npioverL * y)
                       * std::sin(npioverL * z);

    double coscossin = std::cos(npioverL * x) * std::cos(npioverL * y)
                       * std::sin(npioverL * z);

    r = std::sqrt(
        (x - xc) * (x - xc) + (y - yc) * (y - yc) + (z - zc) * (z - zc));

    sol(i, j, k) = sincossin;

    if (vbx.contains(i, j, k))
    {
        rhs(i, j, k) = (1.0 + npioverL * npioverL) * beta(i, j, k) * sincossin;
        alpha(i, j, k) = 1.0;
    }

    // Robin BC
    bool robin_cell = false;
    double sign     = 1.0;
    if (robin_dir == 0 && j >= dlo.y && j <= dhi.y && k >= dlo.z && k <= dhi.z)
    {
        if (i > dhi.x)
        {
            robin_cell = true;
            sign       = -1.0;
        }

        if (i < dlo.x)
        {
            robin_cell = true;
            sign       = 1.0;
        }
    }
    else if (robin_dir == 1 && i >= dlo.x && i <= dhi.x && k >= dlo.z
             && k <= dhi.z)
    {
        robin_cell = (j > dhi.y) || (j < dlo.y);
    }
    else if (robin_dir == 2 && i >= dlo.x && i <= dhi.x && j >= dlo.y
             && j <= dhi.y)
    {
        robin_cell = (k > dhi.z) || (k < dlo.z);
    }
    if (robin_cell)
    {
        robin_a(i, j, k) = 1.0;
        robin_b(i, j, k) = -4.0 / 3.0;

        amrex::Real dphidn;
        if (robin_dir == 0)
        {
            dphidn = -tpi * std::sin(tpi * x) * std::cos(tpi * y)
                         * std::cos(tpi * z)
                     - pi * std::sin(fpi * x) * std::cos(fpi * y)
                           * std::cos(fpi * z);
        }
        else if (robin_dir == 1)
        {
            dphidn = -tpi * std::cos(tpi * x) * std::sin(tpi * y)
                         * std::cos(tpi * z)
                     - pi * std::cos(fpi * x) * std::sin(fpi * y)
                           * std::cos(fpi * z);
        }
        else
        {
            dphidn = -tpi * std::cos(tpi * x) * std::cos(tpi * y)
                         * std::sin(tpi * z)
                     - pi * std::cos(fpi * x) * std::cos(fpi * y)
                           * std::sin(fpi * z);
        }
        if (robin_face == 0) dphidn *= -1.0;
        robin_f(i, j, k) = robin_a(i, j, k) * sol(i, j, k)
                           + robin_b(i, j, k) * npioverL * sign * coscossin;
    }
}

void initProbABecLaplacian(amrex::Geometry& geom, amrex::MultiFab& solution,
    amrex::MultiFab& rhs, amrex::MultiFab& exact_solution,
    amrex::MultiFab& acoef, amrex::MultiFab& bcoef, amrex::MultiFab& robin_a,
    amrex::MultiFab& robin_b, amrex::MultiFab& robin_f)
{
    int robin_dir  = 0;
    int robin_face = 0;

    const auto prob_lo = geom.ProbLoArray();
    const auto prob_hi = geom.ProbHiArray();
    const auto dx      = geom.CellSizeArray();
    const auto dlo     = amrex::lbound(geom.Domain());
    const auto dhi     = amrex::ubound(geom.Domain());
    const auto rdir    = robin_dir;
    const auto rface   = robin_face;
    for (amrex::MFIter mfi(rhs); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx  = mfi.validbox();
        amrex::Box const& gbx = amrex::grow(bx, 1);
        auto const& rhsfab    = rhs.array(mfi);
        auto const& solfab    = solution.array(mfi);
        auto const& acfab     = acoef.array(mfi);
        auto const& bcfab     = bcoef.array(mfi);
        auto const& rafab     = robin_a.array(mfi);
        auto const& rbfab     = robin_b.array(mfi);
        auto const& rffab     = robin_f.array(mfi);
        amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE(
                                    int i, int j, int k) noexcept {
            actual_init_abeclap(i, j, k, rhsfab, solfab, acfab, bcfab, rafab,
                rbfab, rffab, prob_lo, prob_hi, dx, dlo, dhi, bx, rdir, rface);
        });
    }

    amrex::MultiFab::Copy(exact_solution, solution, 0, 0, 1, 0);
    solution.setVal(0.0, 0, 1, amrex::IntVect(0));
}

void initMeshandData(PeleRad::AMRParam const& amrpp, amrex::Geometry& geom,
    amrex::BoxArray& grids, amrex::DistributionMapping& dmap,
    amrex::MultiFab& solution, amrex::MultiFab& rhs,
    amrex::MultiFab& exact_solution, amrex::MultiFab& acoef,
    amrex::MultiFab& bcoef, amrex::MultiFab& robin_a, amrex::MultiFab& robin_b,
    amrex::MultiFab& robin_f)
{
    int const n_cell        = amrpp.n_cell_;
    int const max_grid_size = amrpp.max_grid_size_;

    amrex::RealBox rb(
        { AMREX_D_DECL(-1.0, -1.0, -1.0) }, { AMREX_D_DECL(1.0, 1.0, 1.0) });
    amrex::Array<int, AMREX_SPACEDIM> is_periodic { AMREX_D_DECL(0, 0, 0) };
    amrex::Geometry::Setup(&rb, 0, is_periodic.data());
    amrex::Box domain0(amrex::IntVect { AMREX_D_DECL(0, 0, 0) },
        amrex::IntVect { AMREX_D_DECL(n_cell - 1, n_cell - 1, n_cell - 1) });
    geom.define(domain0);

    grids.define(domain0);
    grids.maxSize(max_grid_size);

    amrex::IntVect ng = amrex::IntVect { 1 };

    dmap.define(grids);
    solution.define(grids, dmap, 1, ng);
    rhs.define(grids, dmap, 1, 0);
    exact_solution.define(grids, dmap, 1, ng);
    acoef.define(grids, dmap, 1, 0);
    bcoef.define(grids, dmap, 1, ng);
    robin_a.define(grids, dmap, 1, ng);
    robin_b.define(grids, dmap, 1, ng);
    robin_f.define(grids, dmap, 1, ng);

    initProbABecLaplacian(geom, solution, rhs, exact_solution, acoef, bcoef,
        robin_a, robin_b, robin_f);
}

double check_norm(amrex::MultiFab const& phi, amrex::MultiFab const& exact)
{
    amrex::MultiFab mf(phi.boxArray(), phi.DistributionMap(), 1, 0);
    amrex::MultiFab::Copy(mf, phi, 0, 0, 1, 0);

    amrex::MultiFab::Subtract(mf, exact, 0, 0, 1, 0);

    double L0norm = mf.norm0();
    double L1norm = mf.norm1();
    std::cout << " L0 norm:" << L0norm << ", L1 norm:" << L1norm << std::endl;

    return L1norm;
}

BOOST_AUTO_TEST_CASE(p1_robin)
{
    amrex::ParmParse pp;
    PeleRad::AMRParam amrpp(pp);
    PeleRad::MLMGParam mlmgpp(pp);

    int const n_cell = amrpp.n_cell_;

    amrex::Geometry geom;
    amrex::BoxArray grids;
    amrex::DistributionMapping dmap;

    amrex::MultiFab solution;
    amrex::MultiFab rhs;
    amrex::MultiFab exact_solution;

    amrex::MultiFab acoef;
    amrex::MultiFab bcoef;
    amrex::MultiFab robin_a;
    amrex::MultiFab robin_b;
    amrex::MultiFab robin_f;

    std::cout << "initialize data ... \n";
    initMeshandData(amrpp, geom, grids, dmap, solution, rhs, exact_solution,
        acoef, bcoef, robin_a, robin_b, robin_f);
    std::cout << "construct the PDE ... \n";
    amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> lobc { AMREX_D_DECL(
        amrex::LinOpBCType::Robin, amrex::LinOpBCType::Dirichlet,
        amrex::LinOpBCType::Neumann) };
    amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> hibc { AMREX_D_DECL(
        amrex::LinOpBCType::Robin, amrex::LinOpBCType::Dirichlet,
        amrex::LinOpBCType::Neumann) };
    PeleRad::POneSingle rte(amrpp, mlmgpp, geom, grids, dmap, solution, rhs,
        acoef, bcoef, lobc, hibc, robin_a, robin_b, robin_f);
    std::cout << "solve the PDE ... \n";
    rte.solve();

    auto eps = check_norm(solution, exact_solution);
    eps /= static_cast<double>(n_cell * n_cell * n_cell);
    std::cout << "normalized L1 norm:" << eps << std::endl;

    // plot results
    amrex::MultiFab plotmf(grids, dmap, 4, 0);
    amrex::MultiFab::Copy(plotmf, solution, 0, 0, 1, 0);
    amrex::MultiFab::Copy(plotmf, rhs, 0, 1, 1, 0);
    amrex::MultiFab::Copy(plotmf, exact_solution, 0, 2, 1, 0);
    amrex::MultiFab::Copy(plotmf, solution, 0, 3, 1, 0);
    amrex::MultiFab::Subtract(plotmf, plotmf, 2, 3, 1, 0);

    auto const plot_file_name = amrpp.plot_file_name_;
    amrex::WriteSingleLevelPlotfile(plot_file_name, plotmf,
        { "phi", "rhs", "exact", "error" }, geom, 0.0, 0);

    BOOST_TEST(eps < 1e-3);
}
