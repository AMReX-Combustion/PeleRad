#ifndef AMR_PARAM_HPP
#define AMR_PARAM_HPP

#include <AMReX_ParmParse.H>
#include <AMReX_Vector.H>
#include <string>

namespace PeleRad
{

struct AMRParam
{
public:
    amrex::ParmParse pp_;

    int max_level_;
    int ref_ratio_;
    int n_cell_;
    int max_grid_size_;
    int prob_type_;
    std::string plot_file_name_;

    AMRParam() = default;

    AMREX_GPU_HOST
    AMRParam(const amrex::ParmParse& pp) : pp_(pp)
    {
        pp_.query("max_level", max_level_);
        pp_.query("ref_ratio", ref_ratio_);
        pp_.query("n_cell", n_cell_);
        pp_.query("max_grid_size", max_grid_size_);
        pp_.query("prob_type", prob_type_);

        AMREX_ALWAYS_ASSERT_WITH_MESSAGE((max_level_ == 0 || max_level_ == 1),
            "max_level has to be either 0 or 1");

        pp.query("plot_file_name", plot_file_name_);
    }
};
}
#endif
