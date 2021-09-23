#ifndef RADIATION_HPP
#define RADIATION_HPP

//#include <AMRParam.hpp>
//#include <MLMGParam.hpp>
#include <AMReX_PlotFileUtil.H>
#include <Constants.hpp>
#include <POneSingle.hpp>
#include <PlanckMean.hpp>
#include <SpectralModels.hpp>

namespace PeleRad
{

class Radiation
{
private:
    //    AMRParam amrpp_;
    //    MLMGParam mlmgpp_;

    //    amrex::ParmParse const& pp_;

    PlanckMean radprop;

    amrex::Geometry const& geom_;
    amrex::BoxArray const& grids_;
    amrex::DistributionMapping const& dmap_;

    amrex::MultiFab solution_;
    amrex::MultiFab rhs_;
    amrex::MultiFab acoef_;
    amrex::MultiFab bcoef_;
    amrex::MultiFab robin_a_;
    amrex::MultiFab robin_b_;
    amrex::MultiFab robin_f_;

    amrex::MultiFab absc_;

    RadComps rc_;

public:
    AMREX_GPU_HOST
    Radiation(amrex::Geometry const& geom, amrex::BoxArray const& grids,
        amrex::DistributionMapping const& dmap, RadComps rc)
        : geom_(geom), grids_(grids), dmap_(dmap), rc_(rc)
    {
        rc_.checkIndices();
        solution_.define(
            grids_, dmap_, 1, 1); // one ghost cell to store boundary conditions
        rhs_.define(grids_, dmap_, 1, 0);
        acoef_.define(grids_, dmap_, 1, 0);
        bcoef_.define(
            grids_, dmap_, 1, 1); // one ghost cell for averaging to faces
        robin_a_.define(grids_, dmap_, 1, 1);
        robin_b_.define(grids_, dmap_, 1, 1);
        robin_f_.define(grids_, dmap_, 1, 1);
        absc_.define(grids_, dmap_, 1, 0);

        solution_.setVal(0.0, 0, 1, amrex::IntVect(0));

        loadSpecModel();
    }

    AMREX_GPU_HOST
    void loadSpecModel()
    {
        std::string data_path;
        data_path = "/ccs/home/gwjgavin/Pele_dev/PeleRad/data/kpDB/";

        radprop.load(data_path);

        std::cout << "The radiative property database is loaded" << std::endl;
    }

    void readRadParams(amrex::ParmParse const& pp) { }

    void updateSpecProp(
        amrex::MFIter const& mfi, amrex::Array4<const amrex::Real> const& Yco2,
        amrex::Array4<const amrex::Real> const& Yh2o,
        amrex::Array4<const amrex::Real> const& Yco,
        amrex::Array4<const amrex::Real> const& T,
        amrex::Array4<const amrex::Real> const& P
#ifdef SOOT_MODEL
        ,
        amrex::Array4<const amrex::Real> const& fv
#endif
    )
    {
        // std::cout << "update radiative properties" << std::endl;
        auto const& kpco2 = radprop.kpco2();
        auto const& kph2o = radprop.kph2o();
        auto const& kpco  = radprop.kpco();

        //        auto const&kpsoot = radprop.kpsoot();
        amrex::Box const& bx  = mfi.validbox();
        amrex::Box const& gbx = mfi.growntilebox(1);

        auto kappa       = absc_.array(mfi);
        auto rhsfab      = rhs_.array(mfi);
        auto alphafab    = acoef_.array(mfi);
        auto betafab     = bcoef_.array(mfi);
        auto robin_a_fab = robin_a_.array(mfi);
        auto robin_b_fab = robin_b_.array(mfi);
        auto robin_f_fab = robin_f_.array(mfi);

        amrex::ParallelFor(
            bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                RadProp::getRadPropGas(
                    i, j, k, Yco2, Yh2o, Yco, T, P, kappa, kpco2, kph2o, kpco);
            });

#ifdef SOOT_MODEL
        auto const& kpsoot = radprop.kpsoot();
        amrex::ParallelFor(
            bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                RadProp::getRadPropSoot(i, j, k, fv, T, kappa, kpsoot);
            });
#endif

        amrex::ParallelFor(
            gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                betafab(i, j, k) = 0.00001;

                if (bx.contains(i, j, k))
                {
                    double ka        = std::max(0.00001, kappa(i, j, k));
                    betafab(i, j, k) = 1.0 / ka;
                    // rhsfab(i, j, k)
                    //    = 4.0 * ka * 5.67e-8 * std::pow(T(i, j, k), 4.0); //si
                    rhsfab(i, j, k)
                        = 4.0 * ka * 5.67e-5 * std::pow(T(i, j, k), 4.0); // cgs
                    alphafab(i, j, k) = ka;
                }

                // Robin BC
                /*
                bool robin_cell = false;
                double sign     = 1.0;
                if (j >= dlo.y && j <= dhi.y && k >= dlo.z && k <= dhi.z)
                {
                    if (i > dhi.x)
                    {
                        robin_cell = true;
                        sign       = -1.0;
                    }

                    if (i < dlo.x)
                    {
                        robin_cell = true;
                        sign       = 1.0;
                    }
                }
                else if (i >= dlo.x && i <= dhi.x && k >= dlo.z && k <= dhi.z)
                {
                    if (j > dhi.y)
                    {
                        robin_cell = true;
                        sign       = -1.0;
                    }
                    if (j < dlo.y)
                    {
                        robin_cell = true;
                        sign       = 1.0;
                    }
                }
                else if (i >= dlo.x && i <= dhi.x && j >= dlo.y && j <= dhi.y)
                {
                    if (k > dhi.z)
                    {
                        robin_cell = true;
                        sign = -1.0;
                    }
                    if (k < dlo.z)
                    {
                        robin_cell = true;
                        sign = 1.0;
                    }
                }

                if (robin_cell)
                {
                    robin_a_fab(i, j, k) = beta(i, j, k);
                    robin_b_fab(i, j, k) = -4.0 / 3.0 * sign;

                    robin_f_fab(i, j, k) = 0.0;
                }
    */
            });
    }

    void evaluateRad(amrex::MultiFab& rad_src)
    {
        // std::cout << "evaluateRad() is called" << std::endl;
        amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> lobc { AMREX_D_DECL(
            amrex::LinOpBCType::Neumann, amrex::LinOpBCType::Periodic,
            amrex::LinOpBCType::Periodic) };
        amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> hibc { AMREX_D_DECL(
            amrex::LinOpBCType::Neumann, amrex::LinOpBCType::Periodic,
            amrex::LinOpBCType::Periodic) };

        amrex::ParmParse pp("pelerad");
        // std::cout << "before the mlmgpp" << std::endl;
        MLMGParam mlmgpp(pp);

        // std::cout << "before the rte constructor" << std::endl;
        POneSingle rte(mlmgpp, geom_, grids_, dmap_, solution_, rhs_, acoef_,
            bcoef_, lobc, hibc, robin_a_, robin_b_, robin_f_);
        rte.solve();
        // write();
        // amrex::Abort();

        rte.calcRadSource(rad_src);
        // write();
    }

    RadComps const readRadIndices() const { return rc_; }

    void write()
    {
        amrex::MultiFab plotmf(grids_, dmap_, 4, 0);
        amrex::MultiFab::Copy(plotmf, solution_, 0, 0, 1, 0);
        amrex::MultiFab::Copy(plotmf, rhs_, 0, 1, 1, 0);
        amrex::MultiFab::Copy(plotmf, acoef_, 0, 2, 1, 0);
        amrex::MultiFab::Copy(plotmf, bcoef_, 0, 3, 1, 0);

        amrex::WriteSingleLevelPlotfile("plotrad", plotmf,
            { "solution", "rhs", "acoef", "bcoef" }, geom_, 0.0, 0);
    }
};

} // namespace PeleRad

#endif
