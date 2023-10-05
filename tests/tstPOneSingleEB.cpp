#include <boost/test/unit_test.hpp>

#define BOOST_TEST_MODULE ponerobinsingle

#include <AMReX_FArrayBox.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_PlotFileUtil.H>
#include <POneSingleEB.hpp>

#include <AMReX_EB2.H>
#include <AMReX_EB2_IF.H>
#include <AMReX_EBCellFlag.H>
#include <AMReX_EBFabFactory.H>

AMREX_GPU_DEVICE AMREX_FORCE_INLINE void actual_init_coefs_eb(int i, int j,
    int k, amrex::Array4<amrex::Real> const& phi,
    amrex::Array4<amrex::Real> const& phi_exact,
    amrex::Array4<amrex::Real> const& rhs,
    amrex::Array4<amrex::Real> const& alpha,
    amrex::Array4<amrex::Real> const& bx, amrex::Array4<amrex::Real> const& by,
    amrex::Array4<amrex::Real> const& bz, amrex::Array4<amrex::Real> const& bb,
    amrex::Array4<amrex::EBCellFlag const> const& flag,
    amrex::Array4<amrex::Real const> const& cent,
    amrex::Array4<amrex::Real const> const& bcent,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
    amrex::Box const& vbx)
{
    amrex::Real const L  = std::sqrt(2.0);
    amrex::Real const n  = 3.0;
    amrex::Real npioverL = n * M_PI / L;

    amrex::Real pioverfour    = M_PI / 4.0;
    amrex::Real cospioverfour = std::cos(pioverfour);
    amrex::Real sinpioverfour = std::sin(pioverfour);

    bx(i, j, k) = 1.0;
    by(i, j, k) = 1.0;
    bz(i, j, k) = 1.0;

    if (vbx.contains(i, j, k))
    {
        if (flag(i, j, k).isCovered())
        {
            rhs(i, j, k) = 0.0;
        }
        else
        {
            amrex::Real x = dx[0] * (i + 0.5) - 0.5;
            amrex::Real y = dx[1] * (j + 0.5) - 0.5;
            amrex::Real z = dx[2] * (k + 0.5) - 0.5;

            /*
                        double sincossin = std::sin(npioverL * x) *
               std::cos(npioverL * y)
                                           * std::sin(npioverL * z);
                        double coscossin = std::cos(npioverL * x) *
               std::cos(npioverL * y)
                                           * std::sin(npioverL * z);*/

            amrex::Real xp = x * cospioverfour + y * sinpioverfour;
            amrex::Real yp = -x * sinpioverfour + y * cospioverfour;

            double sincossin = std::sin(npioverL * xp) * std::cos(npioverL * yp)
                               * std::sin(npioverL * z);

            rhs(i, j, k)
                = (1.0 + npioverL * npioverL) * bx(i, j, k) * sincossin;
            alpha(i, j, k) = 1.0;

            phi(i, j, k)       = 0.0;
            phi_exact(i, j, k) = sincossin;
        }
    }
}

void initProbABecLaplacian(amrex::Geometry& geom,
    std::unique_ptr<amrex::EBFArrayBoxFactory>& factory,
    amrex::MultiFab& solution, amrex::MultiFab& rhs,
    amrex::MultiFab& exact_solution, amrex::MultiFab& acoef,
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& bcoef,
    amrex::MultiFab& bcoef_eb)
{

    amrex::FabArray<amrex::EBCellFlagFab> const& flags
        = factory->getMultiEBCellFlagFab();
    amrex::MultiCutFab const& bcent = factory->getBndryCent();
    amrex::MultiCutFab const& cent  = factory->getCentroid();
    /*
            auto const prob_lo = geom.ProbLoArray();
            auto const prob_hi = geom.ProbHiArray();
            auto const dlo     = amrex::lbound(geom.Domain());
            auto const dhi     = amrex::ubound(geom.Domain());
    */
    auto const dx               = geom.CellSizeArray();
    amrex::Box const& domainbox = geom.Domain();
    exact_solution.setVal(0.0);

    for (amrex::MFIter mfi(rhs); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx  = mfi.validbox();
        amrex::Box const& nbx = amrex::surroundingNodes(bx);
        //        amrex::Box const& gbx = amrex::grow(bx, 1);
        amrex::Array4<amrex::Real> const& phi_arr = solution.array(mfi);
        amrex::Array4<amrex::Real> const& phi_ex_arr
            = exact_solution.array(mfi);
        amrex::Array4<amrex::Real> const& rhs_arr = rhs.array(mfi);

        amrex::Array4<amrex::Real> const& bx_arr = bcoef[0].array(mfi);
        amrex::Array4<amrex::Real> const& by_arr = bcoef[1].array(mfi);
        amrex::Array4<amrex::Real> const& bz_arr = bcoef[2].array(mfi);

        auto fabtyp = flags[mfi].getType(bx);
        if (fabtyp == amrex::FabType::covered)
        {
            std::cout << " amrex::FabType::covered == fabtyp \n";
        }
        else if (fabtyp == amrex::FabType::regular)
        {
            std::cout << " amrex::FabType::regular == fabtyp \n";
        }
        else
        {
            //            std::cout << " amrex::FabType  else \n";
            amrex::Array4<amrex::Real> const& acoef_arr = bcoef_eb.array(mfi);
            amrex::Array4<amrex::Real> const& beb_arr   = acoef.array(mfi);
            amrex::Array4<amrex::EBCellFlag const> const& flag_arr
                = flags.const_array(mfi);
            amrex::Array4<amrex::Real const> const& cent_arr
                = cent.const_array(mfi);
            amrex::Array4<amrex::Real const> const& bcent_arr
                = bcent.const_array(mfi);
            amrex::ParallelFor(nbx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                {
                    actual_init_coefs_eb(i, j, k, phi_arr, phi_ex_arr, rhs_arr,
                        acoef_arr, bx_arr, by_arr, bz_arr, beb_arr, flag_arr,
                        cent_arr, bcent_arr, dx, bx);
                });
        }
    }
}

void initMeshandData(PeleRad::AMRParam const& amrpp,
    PeleRad::MLMGParam const& mlmgpp, amrex::Geometry& geom,
    amrex::BoxArray& grids, amrex::DistributionMapping& dmap,
    std::unique_ptr<amrex::EBFArrayBoxFactory>& factory,
    amrex::MultiFab& solution, amrex::MultiFab& rhs,
    amrex::MultiFab& exact_solution, amrex::MultiFab& acoef,
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& bcoef,
    amrex::MultiFab& bcoef_eb)
{
    int const n_cell        = amrpp.n_cell_;
    int const max_grid_size = amrpp.max_grid_size_;

    amrex::RealBox rb(
        { AMREX_D_DECL(-1.0, -1.0, -1.0) }, { AMREX_D_DECL(1.0, 1.0, 1.0) });
    amrex::Array<int, AMREX_SPACEDIM> is_periodic { AMREX_D_DECL(1, 1, 1) };
    amrex::Geometry::Setup(&rb, 0, is_periodic.data());
    amrex::Box domain0(amrex::IntVect { AMREX_D_DECL(0, 0, 0) },
        amrex::IntVect { AMREX_D_DECL(n_cell - 1, n_cell - 1, n_cell - 1) });
    //    geom.define(domain0);
    geom.define(domain0, rb, amrex::CoordSys::cartesian, is_periodic);

    grids.define(domain0);
    grids.maxSize(max_grid_size);

    // rotated box
    int const max_coarsening_level = mlmgpp.max_coarsening_level_;
    double const la                = std::sqrt(2.0) / 2.0;
    amrex::EB2::BoxIF box({ AMREX_D_DECL(-la, -la, -la * 0.75) },
        { AMREX_D_DECL(la, la, la * 1.25) }, true);
    auto gshop = amrex::EB2::makeShop(amrex::EB2::translate(
        amrex::EB2::rotate(
            amrex::EB2::translate(box, { AMREX_D_DECL(-0.0, -0.0, -0.0) }),
            std::atan(1.0) * 1.0, 2),
        { AMREX_D_DECL(0.0, 0.0, 0.0) }));
    amrex::EB2::Build(gshop, geom, 0, max_coarsening_level);

    amrex::IntVect ng = amrex::IntVect { 1 };

    dmap.define(grids);

    amrex::EB2::IndexSpace const& eb_is = amrex::EB2::IndexSpace::top();
    amrex::EB2::Level const& eb_level   = eb_is.getLevel(geom);
    //  std::unique_ptr<amrex::EBFArrayBoxFactory> factory;
    factory = std::make_unique<amrex::EBFArrayBoxFactory>(eb_level, geom, grids,
        dmap, amrex::Vector<int> { 2, 2, 2 }, amrex::EBSupport::full);

    solution.define(grids, dmap, 1, 1, amrex::MFInfo(), *factory);
    exact_solution.define(grids, dmap, 1, 0, amrex::MFInfo(), *factory);
    rhs.define(grids, dmap, 1, 0, amrex::MFInfo(), *factory);
    acoef.define(grids, dmap, 1, 0, amrex::MFInfo(), *factory);

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        bcoef[idim].define(
            amrex::convert(grids, amrex::IntVect::TheDimensionVector(idim)),
            dmap, 1, 0, amrex::MFInfo(), *factory);
    }

    bcoef_eb.define(grids, dmap, 1, 0, amrex::MFInfo(), *factory);
    bcoef_eb.setVal(1.0);

    solution.setVal(0.0);
    rhs.setVal(0.0);
    acoef.setVal(1.0);
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        bcoef[idim].setVal(1.0);
    }

    initProbABecLaplacian(
        geom, factory, solution, rhs, exact_solution, acoef, bcoef, bcoef_eb);
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

BOOST_AUTO_TEST_CASE(p1_eb)
{
    amrex::ParmParse pp;
    PeleRad::AMRParam amrpp(pp);
    PeleRad::MLMGParam mlmgpp(pp);

    bool const write = true;
    int const n_cell = amrpp.n_cell_;

    amrex::Geometry geom;
    amrex::BoxArray grids;
    amrex::DistributionMapping dmap;

    amrex::MultiFab solution;
    amrex::MultiFab exact_solution;
    amrex::MultiFab rhs;
    amrex::MultiFab acoef;
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> bcoef;
    amrex::MultiFab bcoef_eb;

    amrex::MultiFab robin_a;
    amrex::MultiFab robin_b;
    amrex::MultiFab robin_f;
    /*
        const amrex::EB2::IndexSpace& eb_is = amrex::EB2::IndexSpace::top();
        const amrex::EB2::Level& eb_level   = eb_is.getLevel(geom);
        amrex::BoxList bl                   = eb_level.boxArray().boxList();
        amrex::Box const& domain            = geom.Domain();
        for (amrex::Box b : bl)
        {
            b &= domain;
        }
        grids.define(bl);
    */
    std::unique_ptr<amrex::EBFArrayBoxFactory> factory;
    /*    factory = std::make_unique<amrex::EBFArrayBoxFactory>(eb_level, geom,
       grids, dmap, amrex::Vector<int> { 2, 2, 2 }, amrex::EBSupport::full);

        solution_eb.define(grids, dmap, 1, 0, amrex::MFInfo(), *factory);*/

    // std::cout << "initialize data ... \n";
    initMeshandData(amrpp, mlmgpp, geom, grids, dmap, factory, solution, rhs,
        exact_solution, acoef, bcoef, bcoef_eb);

    std::cout << "construct the PDE ... \n";
    PeleRad::POneSingleEB rte(mlmgpp, geom, grids, dmap, factory, solution, rhs,
        acoef, bcoef, robin_a, robin_b, robin_f);

    // std::cout << "solve the PDE ... \n";
    rte.solve();

    auto eps = check_norm(solution, exact_solution);
    eps /= static_cast<double>(n_cell * n_cell * n_cell);
    std::cout << "n_cell=" << n_cell << ", normalized L1 norm:" << eps
              << std::endl;

    // plot results
    if (write)
    {
        std::cout << "write the results ... \n";
        amrex::MultiFab const& vfrc = factory->getVolFrac();
        amrex::MultiFab plotmf(grids, dmap, 7, 0);
        amrex::MultiFab::Copy(plotmf, solution, 0, 0, 1, 0);
        amrex::MultiFab::Copy(plotmf, exact_solution, 0, 1, 1, 0);
        amrex::MultiFab::Copy(plotmf, rhs, 0, 2, 1, 0);
        amrex::MultiFab::Copy(plotmf, acoef, 0, 3, 1, 0);
        amrex::MultiFab::Copy(plotmf, bcoef[0], 0, 4, 1, 0);
        amrex::MultiFab::Copy(plotmf, bcoef[1], 0, 5, 1, 0);
        amrex::MultiFab::Copy(plotmf, vfrc, 0, 6, 1, 0);
        /*        amrex::MultiFab::Copy(plotmf, solution, 0, 0, 1, 0);
                amrex::MultiFab::Copy(plotmf, rhs, 0, 1, 1, 0);
                amrex::MultiFab::Copy(plotmf, exact_solution, 0, 2, 1,
           0); amrex::MultiFab::Copy(plotmf, solution, 0, 3, 1, 0);
                amrex::MultiFab::Subtract(plotmf, plotmf, 2, 3, 1, 0);

                auto const plot_file_name = amrpp.plot_file_name_;
                amrex::WriteSingleLevelPlotfile(plot_file_name, plotmf,
                    { "phi", "rhs", "exact", "error" }, geom, 0.0, 0);
        */

        auto const plot_file_name = amrpp.plot_file_name_;
        amrex::WriteSingleLevelPlotfile(plot_file_name, plotmf,
            { "phi", "exact", "rhs", "acoef", "bcoefx", "bcoefy", "vfrac" },
            geom, 0.0, 0);

        // for amrvis
        /*
        amrex::writeFabs(solution, "solution");
        amrex::writeFabs(bcoef, "bcoef");
        amrex::writeFabs(robin_a, "robin_a");
        amrex::writeFabs(robin_b, "robin_b");
        amrex::writeFabs(robin_f, "robin_f");
        */
    }

    // BOOST_TEST(eps < 1e-3);
}
