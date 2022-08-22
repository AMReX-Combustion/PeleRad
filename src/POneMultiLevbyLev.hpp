#ifndef PONEMULTILEVBYLEV_HPP
#define PONEMULTILEVBYLEV_HPP

#include <AMRParam.hpp>
#include <AMReX_FArrayBox.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParmParse.H>
#include <MLMGParam.hpp>

namespace PeleRad
{

class POneMultiLevbyLev
{
private:
    MLMGParam mlmgpp_;

public:
    amrex::Vector<amrex::Geometry> const& geom_;
    amrex::Vector<amrex::BoxArray> const& grids_;
    amrex::Vector<amrex::DistributionMapping> const& dmap_;

    amrex::Vector<amrex::MultiFab>& solution_;
    amrex::Vector<amrex::MultiFab> const& rhs_;
    amrex::Vector<amrex::MultiFab> const& acoef_;
    amrex::Vector<amrex::MultiFab> const& bcoef_;

    amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> const& lobc_;
    amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> const& hibc_;

    amrex::Vector<amrex::MultiFab> const& robin_a_;
    amrex::Vector<amrex::MultiFab> const& robin_b_;
    amrex::Vector<amrex::MultiFab> const& robin_f_;

    amrex::Real const ascalar_ = 1.0;
    amrex::Real const bscalar_ = 1.0 / 3.0;

    //    amrex::Vector<std::unique_ptr<amrex::MLABecLaplacian>> mlabec_;
    // amrex::MLABecLaplacian mlabec_;

    //    amrex::Vector<std::unique_ptr<amrex::MLMG>> mlmg_;

    POneMultiLevbyLev() = delete;

    // constructor
    POneMultiLevbyLev(MLMGParam const& mlmgpp,
        int const ref_ratio,
        amrex::Vector<amrex::Geometry> const& geom,
        amrex::Vector<amrex::BoxArray> const& grids,
        amrex::Vector<amrex::DistributionMapping> const& dmap,
        amrex::Vector<amrex::MultiFab>& solution,
        amrex::Vector<amrex::MultiFab> const& rhs,
        amrex::Vector<amrex::MultiFab> const& acoef,
        amrex::Vector<amrex::MultiFab> const& bcoef,
        amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> const& lobc,
        amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> const& hibc,
        amrex::Vector<amrex::MultiFab> const& robin_a,
        amrex::Vector<amrex::MultiFab> const& robin_b,
        amrex::Vector<amrex::MultiFab> const& robin_f)
        : mlmgpp_(mlmgpp),
          geom_(geom),
          grids_(grids),
          dmap_(dmap),
          solution_(solution),
          rhs_(rhs),
          acoef_(acoef),
          bcoef_(bcoef),
          lobc_(lobc),
          hibc_(hibc),
          robin_a_(robin_a),
          robin_b_(robin_b),
          robin_f_(robin_f)
    {
        auto const max_coarsening_level = mlmgpp_.max_coarsening_level_;
        auto const agglomeration        = mlmgpp_.agglomeration_;
        auto const consolidation        = mlmgpp_.consolidation_;
        auto const linop_maxorder       = mlmgpp_.linop_maxorder_;

        amrex::LPInfo info;
        info.setAgglomeration(agglomeration);
        info.setConsolidation(consolidation);
        info.setMaxCoarseningLevel(max_coarsening_level);

        int const solver_level = 0;
        auto const nlevels     = geom_.size();

        for (int ilev = 0; ilev < nlevels; ++ilev)
        {
            auto const& geom    = geom_[ilev];
            auto& solution      = solution_[ilev];
            auto const& acoef   = acoef_[ilev];
            auto const& bcoef   = bcoef_[ilev];
            auto const& robin_a = robin_a_[ilev];
            auto const& robin_b = robin_b_[ilev];
            auto const& robin_f = robin_f_[ilev];

            amrex::MLABecLaplacian mlabeclev(
                { geom }, { grids_[ilev] }, { dmap_[ilev] }, info);
            mlabeclev.setDomainBC(lobc_, hibc_);
            mlabeclev.setScalars(ascalar_, bscalar_);
            mlabeclev.setMaxOrder(linop_maxorder);

            if (ilev > 0)
            {
                mlabeclev.setCoarseFineBC(&solution_[ilev - 1], ref_ratio);
            }

            auto const max_iter       = mlmgpp_.max_iter_;
            auto const max_fmg_iter   = mlmgpp_.max_fmg_iter_;
            auto const verbose        = mlmgpp_.verbose_;
            auto const bottom_verbose = mlmgpp_.bottom_verbose_;
            auto const use_hypre      = mlmgpp_.use_hypre_;

            amrex::MLMG mlmglev(mlabeclev);
            mlmglev.setMaxIter(max_iter);
            mlmglev.setMaxFmgIter(max_fmg_iter);
            mlmglev.setVerbose(verbose);
            mlmglev.setBottomVerbose(bottom_verbose);

            if (use_hypre)
                mlmglev.setBottomSolver(amrex::MLMG::BottomSolver::hypre);

            mlabeclev.setLevelBC(
                solver_level, &solution, &robin_a, &robin_b, &robin_f);

            mlabeclev.setACoeffs(solver_level, acoef);

            amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> face_bcoef;
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
            {
                amrex::BoxArray const& ba = amrex::convert(
                    bcoef.boxArray(), amrex::IntVect::TheDimensionVector(idim));
                face_bcoef[idim].define(ba, bcoef.DistributionMap(), 1, 0);
            }
            amrex::average_cellcenter_to_face(
                GetArrOfPtrs(face_bcoef), bcoef, geom);
            mlabeclev.setBCoeffs(
                solver_level, amrex::GetArrOfConstPtrs(face_bcoef));
            auto const tol_rel = mlmgpp_.reltol_;
            auto const tol_abs = mlmgpp_.abstol_;

            mlmglev.solve({ &solution }, { &rhs_[ilev] }, tol_rel, tol_abs);
        }
    }

    void solve()
    {
        /*        auto const nlevels     = geom_.size();
                int const solver_level = 0;
                //    amrex::MLABecLaplacian mlabec_(geom_, grids_, dmap_,
           info);

                for (int ilev = 0; ilev < nlevels; ++ilev)
                {
                    auto const& geom    = geom_[ilev];
                    auto& solution      = solution_[ilev];
                    auto const& acoef   = acoef_[ilev];
                    auto const& bcoef   = bcoef_[ilev];
                    auto const& robin_a = robin_a_[ilev];
                    auto const& robin_b = robin_b_[ilev];
                    auto const& robin_f = robin_f_[ilev];

                    mlabec_[ilev]->setLevelBC(
                        solver_level, &solution, &robin_a, &robin_b, &robin_f);

                    mlabec_[ilev]->setACoeffs(ilev, acoef);

                    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> face_bcoef;
                    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
                    {
                        amrex::BoxArray const& ba = amrex::convert(
                            bcoef.boxArray(),
           amrex::IntVect::TheDimensionVector(idim));
                        face_bcoef[idim].define(ba, bcoef.DistributionMap(), 1,
           0);
                    }
                    amrex::average_cellcenter_to_face(
                        GetArrOfPtrs(face_bcoef), bcoef, geom);
                    mlabec_[ilev]->setBCoeffs(
                        solver_level, amrex::GetArrOfConstPtrs(face_bcoef));
                    auto const tol_rel = mlmgpp_.reltol_;
                    auto const tol_abs = mlmgpp_.abstol_;

                    mlmg_[ilev]->solve(solution, rhs_[ilev], tol_rel, tol_abs);
                }
        */
    }
};

}
#endif
