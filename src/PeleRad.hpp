#ifndef RADIATION_HPP
#define RADIATION_HPP

namespace PeleRad
{

class Radiation
{
private:
    AMRParam amrpp_;
    MLMGParam mlmgpp_;

    PlanckMean radprop;

    amrex::Vector<amrex::MultiFab> absc;
    amrex::Vector<amrex::MultiFab> acoef;
    amrex::Vector<amrex::MultiFab> bcoef;
    amrex::Vector<amrex::MultiFab> robin_a;
    amrex::Vector<amrex::MultiFab> robin_b;
    amrex::Vector<amrex::MultiFab> robin_c;

public:
    //radiative heat source term
    amrex::Array4<amrex::Real> radsrc;

    constexpr Radiation() = default;

    AMREX_GPU_HOST
    constexpr Radiation(amrex::ParmParse const& pp) 
    {
      amrpp_(pp);
      mlmgpp_(mlmgpp);
      loadSpecModel();    
    }

    void loadSpecModel(){
        std::string data_path;
        data_path = "../../data/kpDB/";

        radprop(data_path);
    }

    void updateSpecProp()
    {
        auto const&kpco2 = radprop.kpco2();
        auto const&kpco2 = radprop.kph2o();
        auto const&kpco = radprop.kpco();
        auto const&kpsoot = radprop.kpsoot();
        for(int ilev = 0; ilev < nlevels; ++ilev)
        {
            for(amrex::MFIter mfi(); mfi.isValid(); ++mfi)
            {
                amrex::Box const& bx = mfi.validbox();
                amrex::Box const& gbx = amrex::grow(bx,1);
                auto const& Yco2 = y_co2[ilev].array(mfi);
                auto const& Yh2o = y_h2o[ilev].array(mfi);
                auto const& Yco = y_co[ilev].array(mfi);
                auto const& fv = soot_fv_rad[ilev].array(mfi);
                auto const& T = temperature[ilev].array(mfi);
                auto const& P = pressure[ilev].array(mfi);
                auto const& kappa = absc[ilev].array(mfi);

                amrex::ParallelFor(
                    bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    getRadPropGas(i,j,k,Yco2,Yh2o, Yco, T, P, kappa, kpco2, kph2o, kpco);
                });

                amrex::ParallelFor(
                    bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    getRadPropSoot(i,j,k, fv, T, kappa, kpsoot);
                });
            }
        }  
    }

    void addRadSrc(
      const amrex::Box& vbox, 
      Array4<const Real> const& Qstate,
      Real const time,
      Real const dt) {}

    void evaluateRad()
};

}  // namespace PeleRad

#endif
