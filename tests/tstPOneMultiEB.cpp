#include <boost/test/unit_test.hpp>

#define BOOST_TEST_MODULE ponerobinsingle

#include <AMReX_FArrayBox.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_PlotFileUtil.H>
#include <POneMultiEB.hpp>

#include <AMReX_EB2.H>
#include <AMReX_EB2_IF.H>
#include <AMReX_EBCellFlag.H>
#include <AMReX_EBFabFactory.H>

AMREX_GPU_DEVICE AMREX_FORCE_INLINE void actual_init_coefs_eb(int i, int j,
    int k, amrex::Array4<amrex::Real> const& phi,
    amrex::Array4<amrex::Real> const& phi_exact,
    amrex::Array4<amrex::Real> const& rhs,
    amrex::Array4<amrex::Real> const& acoef,
    amrex::Array4<amrex::Real> const& bx, amrex::Array4<amrex::Real> const& by,
    amrex::Array4<amrex::Real> const& bz,
    amrex::Array4<amrex::EBCellFlag const> const& flag,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
    amrex::Box const& vbx)
{
    amrex::Real const L  = std::sqrt(2.0);
    amrex::Real const n  = 3.0;
    amrex::Real npioverL = n * M_PI / L;

    amrex::Real pioverfour    = M_PI / 4.0;
    amrex::Real cospioverfour = std::cos(pioverfour);
    amrex::Real sinpioverfour = std::sin(pioverfour);

    if (vbx.contains(i, j, k))
    {
        bx(i, j, k) = 1.0;
        by(i, j, k) = 1.0;
        bz(i, j, k) = 1.0;

        phi(i, j, k) = 0.0;

        if (flag(i, j, k).isCovered())
        {
            rhs(i, j, k)       = 0.0;
            phi_exact(i, j, k) = 0.0;
        }
        else
        {
            amrex::Real x = dx[0] * (i + 0.5) - 0.5;
            amrex::Real y = dx[1] * (j + 0.5) - 0.5;
            amrex::Real z = dx[2] * (k + 0.5) - 0.5;

            amrex::Real xp = x * cospioverfour + y * sinpioverfour;
            amrex::Real yp = -x * sinpioverfour + y * cospioverfour;

            double sincossin = std::sin(npioverL * xp) * std::cos(npioverL * yp)
                               * std::sin(npioverL * z);

            rhs(i, j, k)
                = (1.0 + npioverL * npioverL) * bx(i, j, k) * sincossin;
            acoef(i, j, k) = 1.0;

            phi_exact(i, j, k) = sincossin;
        }
    }
}

void initProbABecLaplacian(amrex::Vector<amrex::Geometry>& geom,
    amrex::Vector<std::unique_ptr<amrex::EBFArrayBoxFactory>>& factory,
    amrex::Vector<amrex::MultiFab>& solution,
    amrex::Vector<amrex::MultiFab>& rhs,
    amrex::Vector<amrex::MultiFab>& exact_solution,
    amrex::Vector<amrex::MultiFab>& acoef,
    amrex::Vector<amrex::Array<amrex::MultiFab, 3>>& bcoef)
{
    int const nlevels = geom.size();
    for (int ilev = 0; ilev < nlevels; ++ilev)
    {
        amrex::FabArray<amrex::EBCellFlagFab> const& flags
            = factory[ilev]->getMultiEBCellFlagFab();
        auto const dx               = geom[ilev].CellSizeArray();
        amrex::Box const& domainbox = geom[ilev].Domain();

        for (amrex::MFIter mfi(rhs[ilev]); mfi.isValid(); ++mfi)
        {
            amrex::Box const& bx  = mfi.validbox();
            amrex::Box const& nbx = amrex::surroundingNodes(bx);
            //        amrex::Box const& gbx = amrex::grow(bx, 1);
            amrex::Array4<amrex::Real> const& phi_arr
                = solution[ilev].array(mfi);
            amrex::Array4<amrex::Real> const& phi_ex_arr
                = exact_solution[ilev].array(mfi);
            amrex::Array4<amrex::Real> const& rhs_arr = rhs[ilev].array(mfi);

            amrex::Array4<amrex::Real> const& bx_arr
                = bcoef[ilev][0].array(mfi);
            amrex::Array4<amrex::Real> const& by_arr
                = bcoef[ilev][1].array(mfi);
            amrex::Array4<amrex::Real> const& bz_arr
                = bcoef[ilev][2].array(mfi);

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
                amrex::Array4<amrex::Real> const& acoef_arr
                    = acoef[ilev].array(mfi);
                amrex::Array4<amrex::EBCellFlag const> const& flag_arr
                    = flags.const_array(mfi);
                // amrex::Array4<amrex::Real const> const& cent_arr
                //     = cent.const_array(mfi);
                // amrex::Array4<amrex::Real const> const& bcent_arr
                //     = bcent.const_array(mfi);

                amrex::ParallelFor(nbx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                    {
                        actual_init_coefs_eb(i, j, k, phi_arr, phi_ex_arr,
                            rhs_arr, acoef_arr, bx_arr, by_arr, bz_arr,
                            flag_arr, dx, bx);
                    });
            }
        }
    }
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

BOOST_AUTO_TEST_CASE(p1_multi_eb)
{
    amrex::ParmParse pp;
    PeleRad::AMRParam amrpp(pp);
    PeleRad::MLMGParam mlmgpp(pp);

    bool const write          = true;
    int const n_cell          = amrpp.n_cell_;
    int const nlevels         = amrpp.max_level_ + 1;
    int const max_level       = amrpp.max_level_;
    int const ref_ratio       = amrpp.ref_ratio_;
    int const composite_solve = mlmgpp.composite_solve_;

    int const max_grid_size = amrpp.max_grid_size_;

    amrex::Vector<amrex::Geometry> geom;
    amrex::Vector<amrex::BoxArray> grids;
    amrex::Vector<amrex::DistributionMapping> dmap;

    amrex::Vector<amrex::MultiFab> solution;
    amrex::Vector<amrex::MultiFab> rhs;
    amrex::Vector<amrex::MultiFab> exact_solution;

    amrex::Vector<amrex::MultiFab> acoef;
    amrex::Vector<amrex::Array<amrex::MultiFab, 3>> bcoef;
    amrex::Vector<amrex::MultiFab> robin_a;
    amrex::Vector<amrex::MultiFab> robin_b;
    amrex::Vector<amrex::MultiFab> robin_f;

    amrex::Vector<std::unique_ptr<amrex::EBFArrayBoxFactory>> factory;

    geom.resize(nlevels);
    grids.resize(nlevels);

    amrex::RealBox rb(
        { AMREX_D_DECL(-1.0, -1.0, -1.0) }, { AMREX_D_DECL(1.0, 1.0, 1.0) });
    amrex::Array<int, AMREX_SPACEDIM> is_periodic { AMREX_D_DECL(1, 1, 1) };
    amrex::Geometry::Setup(&rb, 0, is_periodic.data());
    amrex::Box domain0(amrex::IntVect { AMREX_D_DECL(0, 0, 0) },
        amrex::IntVect { AMREX_D_DECL(n_cell - 1, n_cell - 1, n_cell - 1) });
    amrex::Box domain = domain0;
    for (int ilev = 0; ilev < nlevels; ++ilev)
    {
        geom[ilev].define(domain);
        domain.refine(ref_ratio);
    }

    grids[0].define(domain0);
    grids[0].maxSize(max_grid_size);

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
    amrex::EB2::Build(
        gshop, geom.back(), max_level, max_level + max_coarsening_level);

    // refine grid
    for (int ilev = 1; ilev < nlevels; ++ilev)
    {
        amrex::EB2::IndexSpace const& eb_is = amrex::EB2::IndexSpace::top();
        amrex::EB2::Level const& eb_level   = eb_is.getLevel(geom[ilev]);
        amrex::BoxList bl                   = eb_level.boxArray().boxList();
        amrex::Box const& domain            = geom[ilev].Domain();
        for (auto& b : bl)
        {
            b &= domain;
        }
        grids[ilev].define(bl);
    }

    dmap.resize(nlevels);
    factory.resize(nlevels);
    solution.resize(nlevels);
    exact_solution.resize(nlevels);
    rhs.resize(nlevels);
    acoef.resize(nlevels);
    bcoef.resize(nlevels);
    robin_a.resize(nlevels);
    robin_b.resize(nlevels);
    robin_f.resize(nlevels);

    for (int ilev = 0; ilev < nlevels; ++ilev)
    {
        dmap[ilev].define(grids[ilev]);
        amrex::EB2::IndexSpace const& eb_is = amrex::EB2::IndexSpace::top();
        amrex::EB2::Level const& eb_level   = eb_is.getLevel(geom[ilev]);
        factory[ilev] = std::make_unique<amrex::EBFArrayBoxFactory>(eb_level,
            geom[ilev], grids[ilev], dmap[ilev], amrex::Vector<int> { 2, 2, 2 },
            amrex::EBSupport::full);

        solution[ilev].define(
            grids[ilev], dmap[ilev], 1, 1, amrex::MFInfo(), *factory[ilev]);
        exact_solution[ilev].define(
            grids[ilev], dmap[ilev], 1, 0, amrex::MFInfo(), *factory[ilev]);
        rhs[ilev].define(
            grids[ilev], dmap[ilev], 1, 0, amrex::MFInfo(), *factory[ilev]);
        acoef[ilev].define(
            grids[ilev], dmap[ilev], 1, 0, amrex::MFInfo(), *factory[ilev]);

        for (int idim = 0; idim < 3; ++idim)
        {
            bcoef[ilev][idim].define(
                amrex::convert(
                    grids[ilev], amrex::IntVect::TheDimensionVector(idim)),
                dmap[ilev], 1, 0, amrex::MFInfo(), *factory[ilev]);
        }

        solution[ilev].setVal(0.0);
        rhs[ilev].setVal(0.0);
        acoef[ilev].setVal(1.0);
        for (int idim = 0; idim < 3; ++idim)
        {
            bcoef[ilev][idim].setVal(1.0);
        }
    }

    initProbABecLaplacian(
        geom, factory, solution, rhs, exact_solution, acoef, bcoef);

    //     std::cout << "initialize data ... \n";
    //    initMeshandData(amrpp, mlmgpp, geom, grids, dmap, factory,
    //    solution, rhs,
    //        exact_solution, acoef, bcoef);

    std::cout << "construct the PDE ... \n";
    PeleRad::POneMultiEB rte(mlmgpp, geom, grids, dmap, factory, solution, rhs,
        acoef, bcoef, robin_a, robin_b, robin_f);

    std::cout << "solve the PDE ... \n";
    rte.solve();

    double eps     = 0.0;
    double eps_max = 0.0;
    for (int ilev = 0; ilev < nlevels; ++ilev)
    {
        auto dx          = geom[ilev].CellSize();
        amrex::Real dvol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
        eps              = check_norm(solution[ilev], exact_solution[ilev]);
        eps *= dvol;
        std::cout << "Level=" << ilev << ", normalized L1 norm:" << eps
                  << std::endl;
        if (eps > eps_max) eps_max = eps;
    }

    // plot results
    if (write)
    {
        amrex::Vector<amrex::MultiFab> plotmf(nlevels);
        for (int ilev = 0; ilev < nlevels; ++ilev)
        {
            std::cout << "write the results ... \n";
            amrex::MultiFab const& vfrc = factory[ilev]->getVolFrac();
            plotmf[ilev].define(grids[ilev], dmap[ilev], 8, 0);
            amrex::MultiFab::Copy(plotmf[ilev], solution[ilev], 0, 0, 1, 0);
            amrex::MultiFab::Copy(
                plotmf[ilev], exact_solution[ilev], 0, 1, 1, 0);
            amrex::MultiFab::Copy(plotmf[ilev], rhs[ilev], 0, 2, 1, 0);
            amrex::MultiFab::Copy(plotmf[ilev], acoef[ilev], 0, 3, 1, 0);
            amrex::MultiFab::Copy(plotmf[ilev], bcoef[ilev][0], 0, 4, 1, 0);
            amrex::MultiFab::Copy(plotmf[ilev], bcoef[ilev][1], 0, 5, 1, 0);
            amrex::MultiFab::Copy(plotmf[ilev], bcoef[ilev][2], 0, 6, 1, 0);
            amrex::MultiFab::Copy(plotmf[ilev], vfrc, 0, 7, 1, 0);
        }
        auto const plot_file_name = amrpp.plot_file_name_;
        amrex::WriteMultiLevelPlotfile(plot_file_name, nlevels,
            amrex::GetVecOfConstPtrs(plotmf),
            { "phi", "exact", "rhs", "acoef", "bcoefx", "bcoefy", "bcoefz",
                "vfrac" },
            geom, 0.0, amrex::Vector<int>(nlevels, 0),
            amrex::Vector<amrex::IntVect>(
                nlevels, amrex::IntVect { ref_ratio }));

        // for amrvis
        /*
        amrex::writeFabs(solution, "solution");
        amrex::writeFabs(bcoef, "bcoef");
        amrex::writeFabs(robin_a, "robin_a");
        amrex::writeFabs(robin_b, "robin_b");
        amrex::writeFabs(robin_f, "robin_f");
        */
    }

    BOOST_TEST(eps < 1e-1);
}
