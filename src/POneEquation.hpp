#ifndef PONE_EQUATION_HPP
#define PONE_EQUATION_HPP

#include <AMRParam.hpp>
#include <AMReX_EBFabFactory.H>
#include <AMReX_MLEBABecLap.H>
#include <AMReX_MLMG.H>
#include <AMReX_PlotFileUtil.H>
#include <MLMGParam.hpp>

// EB
#include <AMReX_EB2.H>
#include <AMReX_EB2_IF.H>
#include <AMReX_EBMultiFabUtil.H>
#include <EBFlowerIF.hpp>

#include <MyTest_K.H>

using namespace amrex;

namespace PeleRad
{

class POneEquation
{
private:
    // AMR parameters
    AMRParam amrpp_;

    // MLMG
    MLMGParam mlmgpp_;

    Vector<Geometry> geom_;
    Vector<BoxArray> grids_;
    Vector<DistributionMapping> dmap_;
    Vector<std::unique_ptr<EBFArrayBoxFactory>> factory_;

    Vector<MultiFab> phi_;
    Vector<MultiFab> phiexact_;
    Vector<MultiFab> phieb_;
    Vector<MultiFab> rhs_;
    Vector<MultiFab> acoef_;
    Vector<Array<MultiFab, AMREX_SPACEDIM>> bcoef_;
    Vector<MultiFab> bcoef_eb_;

    AMREX_GPU_HOST
    void initializeEB()
    {
        auto max_level            = amrpp_.max_level_;
        auto max_coarsening_level = mlmgpp_.max_coarsening_level_;

        FlowerIF flower(0.3, 0.15, 6, { AMREX_D_DECL(0.5, 0.5, 0.5) }, true);
#if (AMREX_SPACEDIM == 2)
        auto gshop = EB2::makeShop(flower);
#else
        EB2::PlaneIF planelo({ 0.0, 0.0, 0.1 }, { 0.0, 0.0, -1.0 });
        EB2::PlaneIF planehi({ 0.0, 0.0, 0.9 }, { 0.0, 0.0, 1.0 });
        auto gshop = EB2::makeShop(EB2::makeUnion(flower, planelo, planehi));
#endif
        EB2::Build(
            gshop, geom_.back(), max_level, max_level + max_coarsening_level);
    };

    AMREX_GPU_HOST
    void initGrids()
    {
        auto max_level     = amrpp_.max_level_;
        auto ref_ratio     = amrpp_.ref_ratio_;
        auto n_cell        = amrpp_.n_cell_;
        auto max_grid_size = amrpp_.max_grid_size_;

        int nlevels = max_level + 1;
        geom_.resize(nlevels);
        grids_.resize(nlevels);

        RealBox rb(
            { AMREX_D_DECL(0.0, 0.0, 0.0) }, { AMREX_D_DECL(1.0, 1.0, 1.0) });
        std::array<int, AMREX_SPACEDIM> isperiodic { AMREX_D_DECL(0, 0, 0) };
        Geometry::Setup(&rb, 0, isperiodic.data());
        Box domain0(IntVect { AMREX_D_DECL(0, 0, 0) },
            IntVect { AMREX_D_DECL(n_cell - 1, n_cell - 1, n_cell - 1) });
        Box domain = domain0;
        for (int ilev = 0; ilev < nlevels; ++ilev)
        {
            geom_[ilev].define(domain);
            domain.refine(ref_ratio);
        }

        grids_[0].define(domain0);
        grids_[0].maxSize(max_grid_size);
    };

    AMREX_GPU_HOST
    void addFineGrids()
    {
        auto max_level = amrpp_.max_level_;
        for (int ilev = 1; ilev <= max_level; ++ilev)
        {
            const EB2::IndexSpace& eb_is = EB2::IndexSpace::top();
            const EB2::Level& eb_level   = eb_is.getLevel(geom_[ilev]);
            BoxList bl                   = eb_level.boxArray().boxList();
            const Box& domain            = geom_[ilev].Domain();
            for (Box& b : bl)
            {
                b = domain;
            }
            grids_[ilev].define(bl);
        }
    };

public:
    POneEquation() = default;

    AMREX_GPU_HOST
    POneEquation(const AMRParam& amrpp, const MLMGParam& mlmgpp)
        : amrpp_(amrpp), mlmgpp_(mlmgpp)
    {
        initGrids();

        initializeEB();

        addFineGrids();

        initData();
    };

    AMREX_GPU_HOST
    void initData()
    {
        auto max_level = amrpp_.max_level_;
        int nlevels    = max_level + 1;

        dmap_.resize(nlevels);
        factory_.resize(nlevels);
        phi_.resize(nlevels);
        phiexact_.resize(nlevels);
        phieb_.resize(nlevels);
        rhs_.resize(nlevels);
        acoef_.resize(nlevels);
        bcoef_.resize(nlevels);
        bcoef_eb_.resize(nlevels);

        for (int ilev = 0; ilev < nlevels; ++ilev)
        {
            dmap_[ilev].define(grids_[ilev]);
            const EB2::IndexSpace& eb_is = EB2::IndexSpace::top();
            const EB2::Level& eb_level   = eb_is.getLevel(geom_[ilev]);
            factory_[ilev].reset(new EBFArrayBoxFactory(eb_level, geom_[ilev],
                grids_[ilev], dmap_[ilev], { 2, 2, 2 }, EBSupport::full));

            phi_[ilev].define(
                grids_[ilev], dmap_[ilev], 1, 1, MFInfo(), *factory_[ilev]);
            phiexact_[ilev].define(
                grids_[ilev], dmap_[ilev], 1, 0, MFInfo(), *factory_[ilev]);
            phieb_[ilev].define(
                grids_[ilev], dmap_[ilev], 1, 0, MFInfo(), *factory_[ilev]);
            rhs_[ilev].define(
                grids_[ilev], dmap_[ilev], 1, 0, MFInfo(), *factory_[ilev]);
            acoef_[ilev].define(
                grids_[ilev], dmap_[ilev], 1, 0, MFInfo(), *factory_[ilev]);
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
            {
                bcoef_[ilev][idim].define(
                    amrex::convert(
                        grids_[ilev], IntVect::TheDimensionVector(idim)),
                    dmap_[ilev], 1, 0, MFInfo(), *factory_[ilev]);
            }
            bcoef_eb_[ilev].define(
                grids_[ilev], dmap_[ilev], 1, 0, MFInfo(), *factory_[ilev]);
            bcoef_eb_[ilev].setVal(1.0);

            phi_[ilev].setVal(0.0);
            rhs_[ilev].setVal(0.0);
            acoef_[ilev].setVal(0.0);
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
            {
                bcoef_[ilev][idim].setVal(1.0);
            }

            const auto dx         = geom_[ilev].CellSizeArray();
            const Box& domainbox  = geom_[ilev].Domain();
            const auto lprob_type = amrpp_.prob_type_;

            const FabArray<EBCellFlagFab>& flags
                = factory_[ilev]->getMultiEBCellFlagFab();
            const MultiCutFab& bcent = factory_[ilev]->getBndryCent();
            const MultiCutFab& cent  = factory_[ilev]->getCentroid();

            for (MFIter mfi(phiexact_[ilev]); mfi.isValid(); ++mfi)
            {
                const Box& bx                  = mfi.validbox();
                const Box& nbx                 = amrex::surroundingNodes(bx);
                Array4<Real> const& phi_arr    = phi_[ilev].array(mfi);
                Array4<Real> const& phi_ex_arr = phiexact_[ilev].array(mfi);
                Array4<Real> const& phi_eb_arr = phieb_[ilev].array(mfi);
                Array4<Real> const& rhs_arr    = rhs_[ilev].array(mfi);
                AMREX_D_TERM(
                    Array4<Real> const& bx_arr = bcoef_[ilev][0].array(mfi);
                    , Array4<Real> const& by_arr = bcoef_[ilev][1].array(mfi);
                    , Array4<Real> const& bz_arr = bcoef_[ilev][2].array(mfi););

                auto fabtyp = flags[mfi].getType(bx);
                if (FabType::covered == fabtyp)
                {
                    amrex::ParallelFor(
                        bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                            phi_ex_arr(i, j, k) = 0.0;
                            phi_eb_arr(i, j, k) = 0.0;
                        });
                }
                else if (FabType::regular == fabtyp)
                {
                    amrex::ParallelFor(nbx,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                            mytest_set_phi_reg(i, j, k, phi_ex_arr, rhs_arr,
                                AMREX_D_DECL(bx_arr, by_arr, bz_arr), dx,
                                lprob_type, bx);
                        });
                }
                else
                {
                    Array4<Real> const& beb_arr = bcoef_eb_[ilev].array(mfi);
                    Array4<EBCellFlag const> const& flag_arr
                        = flags.const_array(mfi);
                    Array4<Real const> const& cent_arr = cent.const_array(mfi);
                    Array4<Real const> const& bcent_arr
                        = bcent.const_array(mfi);
                    amrex::ParallelFor(nbx,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                            mytest_set_phi_eb(i, j, k, phi_ex_arr, phi_eb_arr,
                                rhs_arr, AMREX_D_DECL(bx_arr, by_arr, bz_arr),
                                beb_arr, flag_arr, cent_arr, bcent_arr, dx,
                                lprob_type, bx);
                        });
                }

                const Box& gbx = mfi.growntilebox(1);
                if (!domainbox.contains(gbx))
                {
                    amrex::ParallelFor(gbx,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                            mytest_set_phi_boundary(
                                i, j, k, phi_arr, dx, domainbox);
                        });
                }
            }
        }
    };

    AMREX_GPU_HOST
    void solve()
    {
        auto verbose              = mlmgpp_.verbose_;
        auto max_coarsening_level = mlmgpp_.max_coarsening_level_;
        auto agg_grid_size        = mlmgpp_.agg_grid_size_;
        auto con_grid_size        = mlmgpp_.con_grid_size_;
        auto linop_maxorder       = mlmgpp_.linop_maxorder_;
        auto scalars              = amrpp_.scalars_;
        auto max_level            = amrpp_.max_level_;

        if (verbose > 0)
        {
            for (int ilev = 0; ilev <= max_level; ++ilev)
            {
                const MultiFab& vfrc = factory_[ilev]->getVolFrac();
                MultiFab v(vfrc.boxArray(), vfrc.DistributionMap(), 1, 0,
                    MFInfo(), *factory_[ilev]);
                MultiFab::Copy(v, vfrc, 0, 0, 1, 0);
                amrex::EB_set_covered(v, 1.0);
                amrex::Print() << "Level " << ilev
                               << ": vfrc min = " << v.min(0) << std::endl;
            }
        }

        std::array<LinOpBCType, AMREX_SPACEDIM> mlmg_lobc;
        std::array<LinOpBCType, AMREX_SPACEDIM> mlmg_hibc;
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            if (geom_[0].isPeriodic(idim))
            {
                mlmg_lobc[idim] = LinOpBCType::Periodic;
                mlmg_hibc[idim] = LinOpBCType::Periodic;
            }
            else
            {
                mlmg_lobc[idim] = LinOpBCType::Dirichlet;
                mlmg_hibc[idim] = LinOpBCType::Dirichlet;
            }
        }

        LPInfo info;
        info.setMaxCoarseningLevel(max_coarsening_level);
        info.setAgglomerationGridSize(agg_grid_size);
        info.setConsolidationGridSize(con_grid_size);

        for (int ilev = 0; ilev <= max_level; ++ilev)
        {
            phi_[ilev].setVal(0.0, 0, 1, IntVect(0));
        }

        static int ipass = 0;
        ++ipass;

        for (int ilev = 0; ilev <= max_level; ++ilev)
        {
            BL_PROFILE_REGION("LEVEL-SOLVE-lev" + std::to_string(ilev) + "-pass"
                              + std::to_string(ipass));

            MLEBABecLap mleb({ geom_[ilev] }, { grids_[ilev] }, { dmap_[ilev] },
                info, { factory_[ilev].get() });
            mleb.setMaxOrder(linop_maxorder);

            mleb.setDomainBC(mlmg_lobc, mlmg_hibc);

            if (ilev > 0)
            {
                mleb.setCoarseFineBC(&phi_[ilev - 1], 2);
            }
            mleb.setLevelBC(0, &phi_[ilev]);

            mleb.setScalars(scalars[0], scalars[1]);

            mleb.setACoeffs(0, acoef_[ilev]);
            mleb.setBCoeffs(0, amrex::GetArrOfConstPtrs(bcoef_[ilev]));

            if (true)
            { // In this test we assume EB is Dirichlet.
                mleb.setEBDirichlet(0, phieb_[ilev], bcoef_eb_[ilev]);
            }

            auto max_iter        = mlmgpp_.max_iter_;
            auto max_fmg_iter    = mlmgpp_.max_fmg_iter_;
            auto max_bottom_iter = mlmgpp_.max_bottom_iter_;
            auto bottom_reltol   = mlmgpp_.bottom_reltol_;
            auto bottom_verbose  = mlmgpp_.bottom_verbose_;
            auto reltol          = mlmgpp_.reltol_;

            MLMG mlmg(mleb);
            mlmg.setMaxIter(max_iter);
            mlmg.setMaxFmgIter(max_fmg_iter);
            mlmg.setBottomMaxIter(max_bottom_iter);
            mlmg.setBottomTolerance(bottom_reltol);
            mlmg.setVerbose(verbose);
            mlmg.setBottomVerbose(bottom_verbose);
            /*
                        if (use_hypre)
                        {
                            mlmg.setBottomSolver(MLMG::BottomSolver::hypre);
                        }
                        else if (use_petsc)
                        {
                            mlmg.setBottomSolver(MLMG::BottomSolver::petsc);
                        }
            */
            const Real tol_rel = reltol;
            const Real tol_abs = 0.0;
            mlmg.solve({ &phi_[ilev] }, { &rhs_[ilev] }, tol_rel, tol_abs);
        }

        if (verbose > 0)
        {
            auto n_cell = amrpp_.n_cell_;
            for (int ilev = 0; ilev <= max_level; ++ilev)
            {
                MultiFab mf(
                    phi_[ilev].boxArray(), phi_[ilev].DistributionMap(), 1, 0);
                MultiFab::Copy(mf, phi_[ilev], 0, 0, 1, 0);
                MultiFab::Subtract(mf, phiexact_[ilev], 0, 0, 1, 0);

                const MultiFab& vfrc = factory_[ilev]->getVolFrac();

                MultiFab::Multiply(mf, vfrc, 0, 0, 1, 0);

                Real norminf = mf.norm0();
                Real norm1   = mf.norm1()
                             * AMREX_D_TERM((1.0 / n_cell), *(1.0 / n_cell),
                                 *(1.0 / n_cell));
                amrex::Print()
                    << "Level " << ilev << ": weighted max and 1 norms "
                    << norminf << ", " << norm1 << std::endl;
            }
        }
    };

    AMREX_GPU_HOST
    void write()
    {
        auto max_level      = amrpp_.max_level_;
        auto plot_file_name = amrpp_.plot_file_name_;

        bool unittest = true;
        Vector<MultiFab> plotmf(max_level + 1);
        if (unittest)
        {
            for (int ilev = 0; ilev <= max_level; ++ilev)
            {
                const MultiFab& vfrc = factory_[ilev]->getVolFrac();
                plotmf[ilev].define(grids_[ilev], dmap_[ilev], 3, 0);
                MultiFab::Copy(plotmf[ilev], phi_[ilev], 0, 0, 1, 0);
                MultiFab::Copy(plotmf[ilev], phiexact_[ilev], 0, 1, 1, 0);
                MultiFab::Copy(plotmf[ilev], vfrc, 0, 2, 1, 0);
            }
            WriteMultiLevelPlotfile(plot_file_name, max_level + 1,
                amrex::GetVecOfConstPtrs(plotmf), { "phi", "exact", "vfrac" },
                geom_, 0.0, Vector<int>(max_level + 1, 0),
                Vector<IntVect>(max_level, IntVect { 2 }));
        }
        else
        {
            for (int ilev = 0; ilev <= max_level; ++ilev)
            {
                const MultiFab& vfrc = factory_[ilev]->getVolFrac();
                plotmf[ilev].define(grids_[ilev], dmap_[ilev], 5, 0);

                MultiFab::Copy(plotmf[ilev], phi_[ilev], 0, 0, 1, 0);

                MultiFab::Copy(plotmf[ilev], phiexact_[ilev], 0, 1, 1, 0);

                MultiFab::Copy(plotmf[ilev], phi_[ilev], 0, 2, 1, 0);
                MultiFab::Subtract(plotmf[ilev], phiexact_[ilev], 0, 2, 1, 0);

                MultiFab::Copy(plotmf[ilev], phi_[ilev], 0, 3, 1, 0);
                MultiFab::Subtract(plotmf[ilev], phiexact_[ilev], 0, 3, 1, 0);
                MultiFab::Multiply(plotmf[ilev], vfrc, 0, 3, 1, 0);

                MultiFab::Copy(plotmf[ilev], vfrc, 0, 4, 1, 0);
            }
            WriteMultiLevelPlotfile(plot_file_name, max_level + 1,
                amrex::GetVecOfConstPtrs(plotmf),
                { "phi", "exact", "error", "error*vfrac", "vfrac" }, geom_, 0.0,
                Vector<int>(max_level + 1, 0),
                Vector<IntVect>(max_level, IntVect { 2 }));
        }
    };
};
}
#endif
