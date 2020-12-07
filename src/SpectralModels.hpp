#ifndef SPECTRAL_MODELS_HPP
#define SPECTRAL_MODELS_HPP

#include <AMReX.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Gpu.H>

namespace PeleRad
{

namespace RadProp
{
    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    void interpT(const amrex::Real& T, int& TindexL, amrex::Real& weight)
    {
        if (T < 300)
        {
            TindexL = 0;
            weight  = 0.0;
            return;
        }
        if (T > 2800)
        {
            TindexL = 125;
            weight  = 1.0;
            return;
        }

        amrex::Real TindexReal = (T - 300.0) / 20.0;
        amrex::Real TindexInte = floor((T - 300.0) / 20.0);
        TindexL                = static_cast<int>(TindexInte);
        weight                 = TindexReal - TindexInte;
        assert(weight <= 1 && weight > 0);
    }

    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    amrex::Real interpk(const int& TindexL, const amrex::Real& weight,
        const amrex::GpuArray<amrex::Real, 126ul>& k)
    {
        return (1.0 - weight) * k[TindexL] + weight * k[TindexL + 1];
    }

    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    void getRadProp(int i, int j, int k,
        const amrex::Array4<const amrex::Real>& mf,
        const amrex::Array4<const amrex::Real>& temp,
        const amrex::Array4<const amrex::Real>& pressure,
        const amrex::Array4<amrex::Real>& absc,
        const amrex::GpuArray<amrex::Real, 126ul>& kdataco2,
        const amrex::GpuArray<amrex::Real, 126ul>& kdatah2o,
        const amrex::GpuArray<amrex::Real, 126ul>& kdataco)
    {
        int TindexL        = 0;
        amrex::Real weight = 1.0;
        // std::cout << "input=" << temp(i, j, k) << std::endl;
        interpT(temp(i, j, k), TindexL, weight);
        // std::cout << "output=" << TindexL << "," << weight << std::endl;

        amrex::Real kp_co2 = interpk(TindexL, weight, kdataco2);
        amrex::Real kp_h2o = interpk(TindexL, weight, kdatah2o);
        amrex::Real kp_co  = interpk(TindexL, weight, kdataco);

        absc(i, j, k) = mf(i, j, k, 0) * kp_co2 + mf(i, j, k, 1) * kp_h2o
                        + mf(i, j, k, 2) * kp_co;
        absc(i, j, k) *= pressure(i, j, k);
    }
}

}
#endif
