#ifndef MLMG_PARAM_HPP
#define MLMG_PARAM_HPP

#include <AMReX_ParmParse.H>

namespace PeleRad
{
struct MLMGParam
{
public:
    amrex::ParmParse pp_;

    int verbose_;
    int bottom_verbose_;
    int max_iter_;
    int max_fmg_iter_;
    int max_bottom_iter_;
    amrex::Real reltol_;
    amrex::Real abstol_;
    amrex::Real bottom_reltol_;
    int linop_maxorder_;
    int max_coarsening_level_;
    int agg_grid_size_;
    int con_grid_size_;
    int agglomeration_;
    int consolidation_;
    int composite_solve_;
    int fine_level_solve_only_;

    amrex::Vector<amrex::Real> scalars_;

    MLMGParam() = default;

    AMREX_GPU_HOST
    MLMGParam(const amrex::ParmParse& pp) : pp_(pp)
    {
        pp_.query("verbose", verbose_);
        pp_.query("bottom_verbose", bottom_verbose_);
        pp_.query("max_iter", max_iter_);
        pp_.query("max_fmg_iter", max_fmg_iter_);
        pp_.query("max_bottom_iter", max_bottom_iter_);
        pp_.query("reltol", reltol_);
        pp_.query("abstol", abstol_);
        pp_.query("bottom_reltol", bottom_reltol_);
        pp_.query("linop_maxorder", linop_maxorder_);
        pp_.query("max_coarsening_level", max_coarsening_level_);
        pp_.query("agg_grid_size", agg_grid_size_);
        pp_.query("con_grid_size", con_grid_size_);
        pp_.query("agglomeration", agglomeration_);
        pp_.query("consolidation", consolidation_);
        pp_.query("composite_solve", composite_solve_);
        pp_.query("fine_level_solve_only", fine_level_solve_only_);

        scalars_.resize(2);
        scalars_[0] = 1.0;
        scalars_[1] = 1.0 / 3.0;

        pp.queryarr("scalars", scalars_);
    }
};
}
#endif
