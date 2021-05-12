#include <boost/test/unit_test.hpp>

#define BOOST_TEST_MODULE ponerobin

#include <AMRParam.hpp>
#include <AMReX_FArrayBox.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParmParse.H>
#include <MLMGParam.hpp>

namespace PeleRad
{
class POneSingle
{
private:
    AMRParam amrpp_;

    MLMGParam mlmgpp_;

public:
    amrex::Geometry const& geom_;
    amrex::BoxArray const& grids_;
    amrex::DistributionMapping const& dmap_;

    amrex::MultiFab& solution_;
    amrex::MultiFab const& rhs_;
    amrex::MultiFab const& acoef_;
    amrex::MultiFab const& bcoef_;

    amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> const& lobc_;
    amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> const& hibc_;

    amrex::MultiFab const& robin_a_;
    amrex::MultiFab const& robin_b_;
    amrex::MultiFab const& robin_f_;

    amrex::Real const ascalar = 1.0;
    amrex::Real const bscalar = 1.0 / 3.0;

    // constructor
    POneSingle(AMRParam const& amrpp, MLMGParam const& mlmgpp,
        amrex::Geometry const& geom, amrex::BoxArray const& grids,
        amrex::DistributionMapping const& dmap, amrex::MultiFab& solution,
        amrex::MultiFab const& rhs, amrex::MultiFab const& acoef,
        amrex::MultiFab const& bcoef,
        amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> const& lobc,
        amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> const& hibc,
        amrex::MultiFab const& robin_a, amrex::MultiFab const& robin_b,
        amrex::MultiFab const& robin_f)
        : amrpp_(amrpp),
          mlmgpp_(mlmgpp),
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

        auto const& geom  = geom_;
        auto const& grids = grids_;
        auto const& dmap  = dmap_;

        auto& solution      = solution_;
        auto const& rhs     = rhs_;
        auto const& acoef   = acoef_;
        auto const& bcoef   = bcoef_;
        auto const& robin_a = robin_a_;
        auto const& robin_b = robin_b_;
        auto const& robin_f = robin_f_;

        amrex::MLABecLaplacian mlabec({ geom }, { grids }, { dmap },
            amrex::LPInfo().setMaxCoarseningLevel(max_coarsening_level));

        mlabec.setDomainBC(lobc_, hibc_);

        mlabec.setLevelBC(0, &solution, &robin_a, &robin_b, &robin_f);

        mlabec.setScalars(ascalar, bscalar);

        mlabec.setACoeffs(0, acoef);

        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> face_bcoef;
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            const amrex::BoxArray& ba = amrex::convert(
                bcoef.boxArray(), amrex::IntVect::TheDimensionVector(idim));
            face_bcoef[idim].define(ba, bcoef.DistributionMap(), 1, 0);
        }
        amrex::average_cellcenter_to_face(
            GetArrOfPtrs(face_bcoef), bcoef, geom);
        mlabec.setBCoeffs(0, amrex::GetArrOfConstPtrs(face_bcoef));

        amrex::MLMG mlmg(mlabec);
        mlmg.setMaxIter(max_iter);
        mlmg.setMaxFmgIter(max_fmg_iter);
        mlmg.setVerbose(verbose);
        mlmg.setBottomVerbose(bottom_verbose);

        if (use_hypre) mlmg.setBottomSolver(amrex::MLMG::BottomSolver::hypre);

        mlmg.solve({ &solution }, { &rhs }, tol_rel, tol_abs);
    }
};
}
