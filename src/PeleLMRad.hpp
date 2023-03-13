#ifndef RADIATION_HPP
#define RADIATION_HPP

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_PlotFileUtil.H>
#include <Constants.hpp>
#include <POneMulti.hpp>
#include <POneMultiLevbyLev.hpp>
#include <PlanckMean.hpp>
#include <SpectralModels.hpp>

namespace PeleRad
{

class Radiation
{
private:
    MLMGParam mlmgpp_;

    int ref_ratio_;

    PlanckMean radprop;

    amrex::Vector<amrex::Geometry>& geom_;
    amrex::Vector<amrex::BoxArray>& grids_;
    amrex::Vector<amrex::DistributionMapping>& dmap_;

    amrex::Vector<amrex::MultiFab> solution_;
    amrex::Vector<amrex::MultiFab> rhs_;
    amrex::Vector<amrex::MultiFab> acoef_;
    amrex::Vector<amrex::MultiFab> bcoef_;
    amrex::Vector<amrex::MultiFab> robin_a_;
    amrex::Vector<amrex::MultiFab> robin_b_;
    amrex::Vector<amrex::MultiFab> robin_f_;

    amrex::Vector<amrex::MultiFab> absc_;

    RadComps rc_;

    bool composite_solve_;

    std::unique_ptr<POneMulti> rte_;

    std::unique_ptr<POneMultiLevbyLev> rtelevbylev_;

public:
    AMREX_GPU_HOST
    Radiation(amrex::Vector<amrex::Geometry>& geom,
        amrex::Vector<amrex::BoxArray>& grids,
        amrex::Vector<amrex::DistributionMapping>& dmap, RadComps rc,
        amrex::ParmParse const& mlmgpp, int const& ref_ratio)
        : geom_(geom),
          grids_(grids),
          dmap_(dmap),
          rc_(rc),
          mlmgpp_(mlmgpp),
          ref_ratio_(ref_ratio)
    {
        if (amrex::ParallelDescriptor::IOProcessor()) rc_.checkIndices();

        auto const nlevels = geom_.size();

        solution_.resize(nlevels);
        rhs_.resize(nlevels);
        acoef_.resize(nlevels);
        bcoef_.resize(nlevels);
        robin_a_.resize(nlevels);
        robin_b_.resize(nlevels);
        robin_f_.resize(nlevels);
        absc_.resize(nlevels);

        initVars(grids, dmap);
        loadSpecModel();

        composite_solve_ = mlmgpp_.composite_solve_;

        if (composite_solve_)
        {
            rte_ = std::make_unique<POneMulti>(mlmgpp_, geom_, grids_, dmap_,
                solution_, rhs_, acoef_, bcoef_, robin_a_, robin_b_, robin_f_);
        }
        else
        {
            rtelevbylev_ = std::make_unique<POneMultiLevbyLev>(mlmgpp_,
                ref_ratio_, geom_, grids_, dmap_, solution_, rhs_, acoef_,
                bcoef_, robin_a_, robin_b_, robin_f_);
        }
    }

    AMREX_GPU_HOST
    void loadSpecModel()
    {
        std::string data_path;
        // spectral database on OLCF
        data_path = "/ccs/home/gwjgavin/Pele_dev/PeleRad/data/kpDB/";

        radprop.load(data_path);

        amrex::Print() << "The radiative property database is loaded"
                       << "\n";
    }

    void updateSpecProp(amrex::MFIter const& mfi,
        amrex::Array4<const amrex::Real> const& Yco2,
        amrex::Array4<const amrex::Real> const& Yh2o,
        amrex::Array4<const amrex::Real> const& Yco,
        amrex::Array4<const amrex::Real> const& T,
        amrex::Array4<const amrex::Real> const& P
#ifdef PELELM_USE_SOOT
        ,
        amrex::Array4<const amrex::Real> const& fv
#endif
        ,
        int ilev)
    {
        amrex::Print() << "update radiative properties"
                       << "\n";

        auto const& kpco2 = radprop.kpco2();
        auto const& kph2o = radprop.kph2o();
        auto const& kpco  = radprop.kpco();

        amrex::Box const& bx  = mfi.validbox();
        amrex::Box const& gbx = amrex::grow(bx, 1);

        auto const dlo = amrex::lbound(geom_[ilev].Domain());
        auto const dhi = amrex::ubound(geom_[ilev].Domain());

        auto const& kappa       = absc_[ilev].array(mfi);
        auto const& rhsfab      = rhs_[ilev].array(mfi);
        auto const& alphafab    = acoef_[ilev].array(mfi);
        auto const& betafab     = bcoef_[ilev].array(mfi);
        auto const& robin_a_fab = robin_a_[ilev].array(mfi);
        auto const& robin_b_fab = robin_b_[ilev].array(mfi);
        auto const& robin_f_fab = robin_f_[ilev].array(mfi);

        amrex::ParallelFor(
            bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                RadProp::getRadPropGas(
                    i, j, k, Yco2, Yh2o, Yco, T, P, kappa, kpco2, kph2o, kpco);
            });

#ifdef PELELM_USE_SOOT
        auto const& kpsoot = radprop.kpsoot();
        amrex::ParallelFor(
            bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                RadProp::getRadPropSoot(i, j, k, fv, T, kappa, kpsoot);
            });
#endif

        amrex::ParallelFor(
            gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                betafab(i, j, k) = 1.0;

                if (bx.contains(i, j, k))
                {
                    double ka        = std::max(0.01, kappa(i, j, k) * 100);
                    betafab(i, j, k) = 1.0 / ka;

                    rhsfab(i, j, k) = 4.0 * ka * 5.67e-8
                                      * std::pow(T(i, j, k),
                                          4.0); // SI

                    /*rhsfab(i, j, k) = 4.0 * ka * 5.67e-5
                                      * std::pow(T(i, j, k),
                                          4.0);*/ // cgs
                    alphafab(i, j, k) = ka;
                }

                // Robin BC
                bool robin_cell = false;
                if (j >= dlo.y && j <= dhi.y && k >= dlo.z && k <= dhi.z)
                {
                    if (i > dhi.x || i < dlo.x) { robin_cell = true; }
                }
                else if (i >= dlo.x && i <= dhi.x && k >= dlo.z && k <= dhi.z)
                {
                    if (j > dhi.y || j < dlo.y) { robin_cell = true; }
                }
                else if (i >= dlo.x && i <= dhi.x && j >= dlo.y && j <= dhi.y)
                {
                    if (k > dhi.z || k < dlo.z) { robin_cell = true; }
                }

                if (robin_cell)
                {
                    robin_a_fab(i, j, k) = -1.0 / betafab(i, j, k);
                    robin_b_fab(i, j, k) = -2.0 / 3.0;
                    robin_f_fab(i, j, k) = 0.0;
                }
            });
    }

    void initVars(amrex::Vector<amrex::BoxArray> const& grids,
        amrex::Vector<amrex::DistributionMapping> const& dmap)
    {
        amrex::IntVect ng = amrex::IntVect { 1 };
        grids_            = grids;
        dmap_             = dmap;

        for (int ilev = 0; ilev < grids.size(); ++ilev)
        {
            solution_[ilev].define(grids[ilev], dmap[ilev], 1, ng);
            rhs_[ilev].define(grids[ilev], dmap[ilev], 1, 0);
            acoef_[ilev].define(grids[ilev], dmap[ilev], 1, 0);
            bcoef_[ilev].define(grids[ilev], dmap[ilev], 1, ng);
            robin_a_[ilev].define(grids[ilev], dmap[ilev], 1, ng);
            robin_b_[ilev].define(grids[ilev], dmap[ilev], 1, ng);
            robin_f_[ilev].define(grids[ilev], dmap[ilev], 1, ng);
            absc_[ilev].define(grids[ilev], dmap[ilev], 1, 0);

            solution_[ilev].setVal(0.0, 0, 1, ng);
            bcoef_[ilev].setVal(1.0, 0, 1, ng);
        }
    }

    void evaluateRad()
    {
        auto const nlevels = geom_.size();

        if (composite_solve_) { rte_->solve(); }
        else
        {
            rtelevbylev_->solve();
        }
    }

    void calcRadSource(amrex::MFIter const& mfi,
        amrex::Array4<amrex::Real> const& radfab, int ilev)
    {
        amrex::Box const& bx = mfi.validbox();
        auto const& rhsfab   = rhs_[ilev].array(mfi);
        auto const& solfab   = solution_[ilev].array(mfi);
        auto const& acfab    = acoef_[ilev].array(mfi);

        amrex::ParallelFor(
            bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                radfab(i, j, k)
                    += acfab(i, j, k) * solfab(i, j, k) - rhsfab(i, j, k);
            });
    }

    RadComps const readRadIndices() const { return rc_; }

    amrex::Vector<amrex::MultiFab> const& G() { return solution_; }

    amrex::Vector<amrex::MultiFab> const& kappa() { return acoef_; }

    amrex::Vector<amrex::MultiFab> const& emis() { return rhs_; }

    amrex::Vector<amrex::BoxArray> const& grids() { return grids_; }
};

} // namespace PeleRad

#endif
