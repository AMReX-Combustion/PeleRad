#include <boost/test/unit_test.hpp>

#define BOOST_TEST_MODULE 1DHomo

#include <AMReX_PlotFileUtil.H>
#include <POneSingle.hpp>

AMREX_GPU_DEVICE AMREX_FORCE_INLINE void actual_init_coefs(int i, int j, int k,
    amrex::Array4<amrex::Real> const& rhs,
    amrex::Array4<amrex::Real> const& sol,
    amrex::Array4<amrex::Real> const& alpha,
    amrex::Array4<amrex::Real> const& beta,
    amrex::Array4<amrex::Real> const& robin_a,
    amrex::Array4<amrex::Real> const& robin_b,
    amrex::Array4<amrex::Real> const& robin_f,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& prob_lo,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& prob_hi,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
    amrex::Dim3 const& dlo, amrex::Dim3 const& dhi, amrex::Box const& vbx)
{
    // beta(i, j, k) = 0.1;
    // beta(i, j, k) = 1.0;
    beta(i, j, k) = 2.0;
    // beta(i, j, k) = 1000.0;

    if (vbx.contains(i, j, k))
    {
        alpha(i, j, k) = 1.0 / beta(i, j, k);
        rhs(i, j, k)   = alpha(i, j, k) * 1.0;
    }

    // Robin BC
    bool robin_cell = false;

    if (j >= dlo.y && j <= dhi.y && k >= dlo.z && k <= dhi.z)
    {
        if (i > dhi.x || i < dlo.x) { robin_cell = true; }
    }

    if (robin_cell)
    {
        robin_a(i, j, k) = -1.0 / beta(i, j, k);
        robin_b(i, j, k) = -2.0 / 3.0;
        robin_f(i, j, k) = 0.0;
    }
}

void initProbABecLaplacian(amrex::Geometry& geom, amrex::MultiFab& solution,
    amrex::MultiFab& rhs, amrex::MultiFab& acoef, amrex::MultiFab& bcoef,
    amrex::MultiFab& robin_a, amrex::MultiFab& robin_b,
    amrex::MultiFab& robin_f)
{
    auto const prob_lo = geom.ProbLoArray();
    auto const prob_hi = geom.ProbHiArray();
    auto const dx      = geom.CellSizeArray();
    auto const dlo     = amrex::lbound(geom.Domain());
    auto const dhi     = amrex::ubound(geom.Domain());
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
        amrex::ParallelFor(gbx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                actual_init_coefs(i, j, k, rhsfab, solfab, acfab, bcfab, rafab,
                    rbfab, rffab, prob_lo, prob_hi, dx, dlo, dhi, bx);
            });
    }

    solution.setVal(0.0, 0, 1, amrex::IntVect(0));
}

void initMeshandData(PeleRad::AMRParam const& amrpp, amrex::Geometry& geom,
    amrex::BoxArray& grids, amrex::DistributionMapping& dmap,
    amrex::MultiFab& solution, amrex::MultiFab& rhs, amrex::MultiFab& acoef,
    amrex::MultiFab& bcoef, amrex::MultiFab& robin_a, amrex::MultiFab& robin_b,
    amrex::MultiFab& robin_f)
{
    int const n_cell        = amrpp.n_cell_;
    int const max_grid_size = amrpp.max_grid_size_;

    amrex::RealBox rb(
        { AMREX_D_DECL(0.0, 0.0, 0.0) }, { AMREX_D_DECL(1.0, 0.1, 0.1) });
    amrex::Array<int, AMREX_SPACEDIM> is_periodic { AMREX_D_DECL(0, 1, 1) };
    amrex::Geometry::Setup(&rb, 0, is_periodic.data());
    amrex::Box domain0(amrex::IntVect { AMREX_D_DECL(0, 0, 0) },
        amrex::IntVect { AMREX_D_DECL(n_cell - 1, 0, 0) });
    geom.define(domain0);

    grids.define(domain0);
    grids.maxSize(max_grid_size);

    amrex::IntVect ng = amrex::IntVect { 1 };

    dmap.define(grids);
    solution.define(grids, dmap, 1, ng);
    rhs.define(grids, dmap, 1, 0);
    acoef.define(grids, dmap, 1, 0);
    bcoef.define(grids, dmap, 1, ng);
    robin_a.define(grids, dmap, 1, ng);
    robin_b.define(grids, dmap, 1, ng);
    robin_f.define(grids, dmap, 1, ng);

    initProbABecLaplacian(
        geom, solution, rhs, acoef, bcoef, robin_a, robin_b, robin_f);
}

BOOST_AUTO_TEST_CASE(POne1D)
{
    amrex::ParmParse pp;
    PeleRad::AMRParam amrpp(pp);
    PeleRad::MLMGParam mlmgpp(pp);

    bool const write = false;
    int const n_cell = amrpp.n_cell_;

    amrex::Geometry geom;
    amrex::BoxArray grids;
    amrex::DistributionMapping dmap;

    amrex::MultiFab solution;
    amrex::MultiFab rhs;

    amrex::MultiFab acoef;
    amrex::MultiFab bcoef;
    amrex::MultiFab robin_a;
    amrex::MultiFab robin_b;
    amrex::MultiFab robin_f;

    std::cout << "initialize data ... \n";
    initMeshandData(amrpp, geom, grids, dmap, solution, rhs, acoef, bcoef,
        robin_a, robin_b, robin_f);
    std::cout << "construct the PDE ... \n";
    amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> lobc { AMREX_D_DECL(
        amrex::LinOpBCType::Robin, amrex::LinOpBCType::Periodic,
        amrex::LinOpBCType::Periodic) };
    amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> hibc { AMREX_D_DECL(
        amrex::LinOpBCType::Robin, amrex::LinOpBCType::Periodic,
        amrex::LinOpBCType::Periodic) };

    PeleRad::POneSingle rte(mlmgpp, geom, grids, dmap, solution, rhs, acoef,
        bcoef, lobc, hibc, robin_a, robin_b, robin_f);
    std::cout << "solve the PDE ... \n";
    rte.solve();

    // plot results
    if (write)
    {
        std::cout << "write the results ... \n";
        amrex::MultiFab plotmf(grids, dmap, 2, 0);
        amrex::MultiFab::Copy(plotmf, solution, 0, 0, 1, 0);
        amrex::MultiFab::Copy(plotmf, rhs, 0, 1, 1, 0);

        auto const plot_file_name = amrpp.plot_file_name_;
        amrex::WriteSingleLevelPlotfile(
            plot_file_name, plotmf, { "phi", "rhs" }, geom, 0.0, 0);
    }

    BOOST_TEST(1);
}
