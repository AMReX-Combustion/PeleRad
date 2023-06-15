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
    bool use_hypre_;
    amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> lobc_;
    amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> hibc_;
    std::string kppath_;

    AMREX_GPU_HOST
    MLMGParam(const amrex::ParmParse& pp) : pp_(pp)
    {
        pp_.query("kppath", kppath_);
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
        pp_.query("use_hypre", use_hypre_);

        amrex::Vector<std::string> lo_bc_char_(AMREX_SPACEDIM);
        amrex::Vector<std::string> hi_bc_char_(AMREX_SPACEDIM);
        pp_.getarr("lo_bc", lo_bc_char_, 0, AMREX_SPACEDIM);
        pp_.getarr("hi_bc", hi_bc_char_, 0, AMREX_SPACEDIM);

        std::map<std::string, amrex::LinOpBCType> bctype;
        bctype["Dirichlet"] = amrex::LinOpBCType::Dirichlet;
        bctype["Periodic"]  = amrex::LinOpBCType::Periodic;
        bctype["Neumann"]   = amrex::LinOpBCType::Neumann;
        bctype["Robin"]     = amrex::LinOpBCType::Robin;

        amrex::Print() << "Define radiation BC:"
                       << "\n";

        for (int idim = 0; idim < AMREX_SPACEDIM; idim++)
        {
            lobc_[idim] = bctype[lo_bc_char_[idim]];
            hibc_[idim] = bctype[hi_bc_char_[idim]];

            amrex::Print() << "lobc[" << idim << "]=" << lobc_[idim] << ", ";
            amrex::Print() << "hibc[" << idim << "]=" << hibc_[idim] << "\n";
        }
    }
};

}

#endif
