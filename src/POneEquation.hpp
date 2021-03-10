#ifndef PONE_EQUATION_HPP
#define PONE_EQUATION_HPP

#include <AMRParam.hpp>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_PlotFileUtil.H>
#include <MLMGParam.hpp>

using namespace amrex;

namespace PeleRad
{

class POneEquation
{
private:
    // AMR parameters
    AMRParam amrpp_;

    // MLMG parameters
    MLMGParam mlmgpp_;

    Vector<Geometry> geom_;
    Vector<BoxArray> grids_;
    Vector<DistributionMapping> dmap_;

    Vector<MultiFab> phi_;
    Vector<MultiFab> phiexact_;
    Vector<MultiFab> rhs_;
    Vector<MultiFab> acoef_;
    Vector<MultiFab> bcoef_;

public:
    POneEquation() = default;

    AMREX_GPU_HOST
    POneEquation(const AMRParam& amrpp, const MLMGParam& mlmgpp,
        Vector<Geometry> const geom, Vector<BoxArray> const grids,
        Vector<DistributionMapping> const dmap)
        : amrpp_(amrpp),
          mlmgpp_(mlmgpp),
          geom_(geom),
          grids_(grids),
          dmap_(dmap) {};

    AMREX_GPU_HOST
    void solve(Vector<MultiFab>& soln, Vector<MultiFab> const& alpha,
        Vector<MultiFab> const& beta, Vector<MultiFab> const& rhs,
        Vector<MultiFab> const& exact)
    {
        auto verbose               = mlmgpp_.verbose_;
        auto bottom_verbose        = mlmgpp_.bottom_verbose_;
        auto composite_solve       = mlmgpp_.composite_solve_;
        auto fine_level_solve_only = mlmgpp_.fine_level_solve_only_;
        auto max_iter              = mlmgpp_.max_iter_;
        auto max_fmg_iter          = mlmgpp_.max_fmg_iter_;
        auto max_coarsening_level  = mlmgpp_.max_coarsening_level_;
        auto agglomeration         = mlmgpp_.agglomeration_;
        auto consolidation         = mlmgpp_.consolidation_;
        auto linop_maxorder        = mlmgpp_.linop_maxorder_;
        auto rel_tol               = mlmgpp_.reltol_;
        auto abs_tol               = mlmgpp_.abstol_;
        auto ref_ratio             = amrpp_.ref_ratio_;
        auto use_hypre             = mlmgpp_.use_hypre_;

        /*
        std::cout << "composite_solve=" << composite_solve
                  << ",fine_level_solve_only=" << fine_level_solve_only
                  << ",max_iter=" << max_iter
                  << ",max_fmg_iter=" << max_fmg_iter
                  << ",max_coarsening_level  =" << max_coarsening_level
                  << ",agglomeration =" << agglomeration
                  << ",consolidation =" << consolidation
                  << ",linop_maxorder=" << linop_maxorder
                  << ",rel_tol=" << rel_tol << ",abs_tol=" << abs_tol;
        */

        std::array<LinOpBCType, AMREX_SPACEDIM> mlmg_lobc;
        std::array<LinOpBCType, AMREX_SPACEDIM> mlmg_hibc;

        mlmg_lobc[0] = LinOpBCType::Neumann;
        mlmg_hibc[0] = LinOpBCType::Neumann;

        mlmg_lobc[1] = LinOpBCType::Dirichlet;
        mlmg_hibc[1] = LinOpBCType::Dirichlet;

        mlmg_lobc[2] = LinOpBCType::Neumann;
        mlmg_hibc[2] = LinOpBCType::Neumann;

        LPInfo info;
        info.setAgglomeration(agglomeration);
        info.setConsolidation(consolidation);
        info.setMaxCoarseningLevel(max_coarsening_level);

        const int nlevels = geom_.size();

        double const a = 1.0;
        double const b = 1.0 / 3.0;

        if (composite_solve)
        {
            Vector<MultiFab*> psoln;
            Vector<MultiFab const*> prhs;

            for (int ilev = 0; ilev < nlevels; ++ilev)
            {
                psoln.push_back(&(soln[ilev]));
                prhs.push_back(&(rhs[ilev]));
            }

            MLABecLaplacian mlabec(geom_, grids_, dmap_, info);
            mlabec.setMaxOrder(linop_maxorder);
            // BC
            mlabec.setDomainBC({ mlmg_lobc[0], mlmg_lobc[1], mlmg_lobc[2] },
                { mlmg_hibc[0], mlmg_hibc[1], mlmg_hibc[2] });
            for (int ilev = 0; ilev < nlevels; ++ilev)
            {
                mlabec.setLevelBC(ilev, psoln[ilev]);
            }
            mlabec.setScalars(a, b);
            for (int ilev = 0; ilev < nlevels; ++ilev)
            {
                mlabec.setACoeffs(ilev, alpha[ilev]);
                std::array<MultiFab, AMREX_SPACEDIM> bcoefs;
                for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
                {
                    const BoxArray& ba = amrex::convert(beta[ilev].boxArray(),
                        IntVect::TheDimensionVector(idim));
                    bcoefs[idim].define(ba, beta[ilev].DistributionMap(), 1, 0);
                }
                amrex::average_cellcenter_to_face(
                    amrex::GetArrOfPtrs(bcoefs), beta[ilev], geom_[ilev]);
                mlabec.setBCoeffs(ilev, amrex::GetArrOfConstPtrs(bcoefs));
            }

            MLMG mlmg(mlabec);
            mlmg.setMaxIter(max_iter);
            mlmg.setMaxFmgIter(max_fmg_iter);

            if (use_hypre) mlmg.setBottomSolver(MLMG::BottomSolver::hypre);

            mlmg.setVerbose(verbose);
            mlmg.setBottomVerbose(bottom_verbose);
            mlmg.solve(psoln, prhs, rel_tol, abs_tol);
        }
        else
        {
            const int levbegin = (fine_level_solve_only) ? nlevels - 1 : 0;
            for (int ilev = 0; ilev < levbegin; ++ilev)
            {
                MultiFab::Copy(soln[ilev], exact[ilev], 0, 0, 1, 0);
            }
            for (int ilev = levbegin; ilev < nlevels; ++ilev)
            {
                MLABecLaplacian mlabec({ geom_[ilev] },
                    { soln[ilev].boxArray() }, { soln[ilev].DistributionMap() },
                    info);
                mlabec.setMaxOrder(linop_maxorder);
                mlabec.setDomainBC({ mlmg_lobc[0], mlmg_lobc[1], mlmg_lobc[2] },
                    { mlmg_hibc[0], mlmg_hibc[1], mlmg_hibc[2] });
                int const solver_level = 0;
                if (ilev > 0)
                {
                    mlabec.setCoarseFineBC(&soln[ilev - 1], ref_ratio);
                }
                mlabec.setLevelBC(solver_level, &soln[ilev]);

                mlabec.setScalars(a, b);
                mlabec.setACoeffs(solver_level, alpha[ilev]);

                std::array<MultiFab, AMREX_SPACEDIM> bcoefs;
                for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
                {
                    const BoxArray& ba = amrex::convert(beta[ilev].boxArray(),
                        IntVect::TheDimensionVector(idim));
                    bcoefs[idim].define(ba, beta[ilev].DistributionMap(), 1, 0);
                }
                amrex::average_cellcenter_to_face(
                    amrex::GetArrOfPtrs(bcoefs), beta[ilev], geom_[ilev]);
                mlabec.setBCoeffs(
                    solver_level, amrex::GetArrOfConstPtrs(bcoefs));

                MLMG mlmg(mlabec);
                mlmg.setMaxIter(max_iter);
                mlmg.setMaxFmgIter(max_fmg_iter);

                if (use_hypre) mlmg.setBottomSolver(MLMG::BottomSolver::hypre);

                mlmg.setVerbose(verbose);
                mlmg.setBottomVerbose(bottom_verbose);

                mlmg.solve({ &soln[ilev] }, { &rhs[ilev] }, rel_tol, abs_tol);
            }
        }
    };

    AMREX_GPU_HOST
    void write(Vector<MultiFab>& soln, Vector<MultiFab> const& alpha,
        Vector<MultiFab> const& beta, Vector<MultiFab> const& rhs,
        Vector<MultiFab> const& exact, bool const unittest)
    {
        auto max_level      = amrpp_.max_level_;
        auto plot_file_name = amrpp_.plot_file_name_;
        auto ref_ratio      = amrpp_.ref_ratio_;

        Vector<MultiFab> plotmf(max_level + 1);

        if (!unittest)
        {
            for (int ilev = 0; ilev < max_level; ++ilev)
            {
                plotmf[ilev].define(grids_[ilev], dmap_[ilev], 3, 0);
                MultiFab::Copy(plotmf[ilev], soln[ilev], 0, 0, 1, 0);
                // MultiFab::Copy(plotmf[ilev], alpha[ilev], 0, 1, 1, 0);
                // MultiFab::Copy(plotmf[ilev], beta[ilev], 0, 2, 1, 0);
                MultiFab::Copy(plotmf[ilev], rhs[ilev], 0, 1, 1, 0);
                MultiFab::Copy(plotmf[ilev], exact[ilev], 0, 2, 1, 0);
                /// MultiFab::Copy(plotmf[ilev], beta[ilev], 0, 4, 1, 0);
                // MultiFab::Copy(plotmf[ilev], rhs[ilev], 0, 5, 1, 0);
            }
            WriteMultiLevelPlotfile(plot_file_name, max_level + 1,
                amrex::GetVecOfConstPtrs(plotmf),
                //{ "phi", "exact", "absolute error", "alpha", "beta", "rhs" },
                { "phi", "rhs", "mms" }, geom_, 0.0,
                Vector<int>(max_level + 1, 0),
                Vector<IntVect>(max_level,
                    IntVect { AMREX_D_DECL(ref_ratio, ref_ratio, ref_ratio) }));
        }
    }
};
}
#endif
