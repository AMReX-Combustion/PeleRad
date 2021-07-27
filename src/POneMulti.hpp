#ifndef PONEMULTI_HPP
#define PONEMULTI_HPP

#include <AMRParam.hpp>
#include <AMReX_FArrayBox.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParmParse.H>
#include <MLMGParam.hpp>

namespace PeleRad
{

class POneMulti
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

    amrex::Real const ascalar = 1.0;
    amrex::Real const bscalar = 1.0 / 3.0;

    constexpr POneMulti() = default;

    // constructor
    POneMulti(MLMGParam const& mlmgpp,
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
          robin_f_(robin_f) {};

    void solve()
    {
        auto const max_coarsening_level = mlmgpp_.max_coarsening_level_;
        auto const max_iter             = mlmgpp_.max_iter_;
        auto const max_fmg_iter         = mlmgpp_.max_fmg_iter_;
        auto const verbose              = mlmgpp_.verbose_;
        auto const bottom_verbose       = mlmgpp_.bottom_verbose_;
        auto const tol_rel              = mlmgpp_.reltol_;
        auto const tol_abs              = mlmgpp_.abstol_;
        auto const use_hypre            = mlmgpp_.use_hypre_;

        auto const nlevels = geom_.size();

        amrex::MLABecLaplacian mlabec({ geom_ }, { grids_ }, { dmap_ },
            amrex::LPInfo().setMaxCoarseningLevel(max_coarsening_level));

        mlabec.setDomainBC(lobc_, hibc_);

        mlabec.setScalars(ascalar, bscalar);

        for (int ilev = 0; ilev < nlevels; ++ilev)
        {
            auto const& geom    = geom_[ilev];
            auto& solution      = solution_[ilev];
            auto const& acoef   = acoef_[ilev];
            auto const& bcoef   = bcoef_[ilev];
            auto const& robin_a = robin_a_[ilev];
            auto const& robin_b = robin_b_[ilev];
            auto const& robin_f = robin_f_[ilev];

            mlabec.setLevelBC(ilev, &solution, &robin_a, &robin_b, &robin_f);

            mlabec.setACoeffs(ilev, acoef);

            amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> face_bcoef;
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
            {
                amrex::BoxArray const& ba = amrex::convert(
                    bcoef.boxArray(), amrex::IntVect::TheDimensionVector(idim));
                face_bcoef[idim].define(ba, bcoef.DistributionMap(), 1, 0);
            }
            amrex::average_cellcenter_to_face(
                GetArrOfPtrs(face_bcoef), bcoef, geom);
            mlabec.setBCoeffs(ilev, amrex::GetArrOfConstPtrs(face_bcoef));
        }

        amrex::MLMG mlmg(mlabec);
        mlmg.setMaxIter(max_iter);
        mlmg.setMaxFmgIter(max_fmg_iter);
        mlmg.setVerbose(verbose);
        mlmg.setBottomVerbose(bottom_verbose);

        if (use_hypre) mlmg.setBottomSolver(amrex::MLMG::BottomSolver::hypre);

        mlmg.solve(
            GetVecOfPtrs(solution_), GetVecOfConstPtrs(rhs_), tol_rel, tol_abs);
    }
};

}

#endif
