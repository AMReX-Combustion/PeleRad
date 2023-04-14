#ifndef PLANCK_MEAN_HPP
#define PLANCK_MEAN_HPP

#include <filesystem>
#include <fstream>

#include <AMReX.H>
#include <AMReX_Gpu.H>

namespace PeleRad
{

class PlanckMean
{
private:
    amrex::GpuArray<amrex::Real, 126ul> kpco2_;
    amrex::GpuArray<amrex::Real, 126ul> kph2o_;
    amrex::GpuArray<amrex::Real, 126ul> kpco_;
    amrex::GpuArray<amrex::Real, 126ul> kpsoot_;

public:
    AMREX_GPU_HOST
    PlanckMean() { }

    AMREX_GPU_HOST
    PlanckMean(std::string data_path) { load(data_path); }

    AMREX_GPU_HOST
    void load(std::string data_path)
    {
        std::filesystem::path kplco2(data_path + "kpl_co2.dat");
        std::filesystem::path kplh2o(data_path + "kpl_h2o.dat");
        std::filesystem::path kplco(data_path + "kpl_co.dat");
        std::filesystem::path kplsoot(data_path + "kpl_soot.dat");

        std::filesystem::exists(kplco2);
        std::filesystem::exists(kplh2o);
        std::filesystem::exists(kplco);
        std::filesystem::exists(kplsoot);

        std::ifstream dataco2(kplco2);
        std::ifstream datah2o(kplh2o);
        std::ifstream dataco(kplco);
        std::ifstream datasoot(kplsoot);

        size_t i_co2 = 0;
        for (float T_temp, kp_temp; dataco2 >> T_temp >> kp_temp;)
        {
            kpco2_[i_co2] = kp_temp;
            i_co2++;
        }

        size_t i_h2o = 0;
        for (float T_temp, kp_temp; datah2o >> T_temp >> kp_temp;)
        {
            kph2o_[i_h2o] = kp_temp;
            i_h2o++;
        }

        size_t i_co = 0;
        for (float T_temp, kp_temp; dataco >> T_temp >> kp_temp;)
        {
            kpco_[i_co] = kp_temp;
            i_co++;
        }

        size_t i_soot = 0;
        for (float T_temp, kp_temp; datasoot >> T_temp >> kp_temp;)
        {
            kpsoot_[i_soot] = kp_temp;
            i_soot++;
        }

        dataco2.close();
        datah2o.close();
        dataco.close();
        datasoot.close();
    }

    AMREX_GPU_HOST_DEVICE
    const amrex::GpuArray<amrex::Real, 126ul>& kpco2() const { return kpco2_; }

    AMREX_GPU_HOST_DEVICE
    const amrex::GpuArray<amrex::Real, 126ul>& kph2o() const { return kph2o_; }

    AMREX_GPU_HOST_DEVICE
    const amrex::GpuArray<amrex::Real, 126ul>& kpco() const { return kpco_; }

    AMREX_GPU_HOST_DEVICE
    const amrex::GpuArray<amrex::Real, 126ul>& kpsoot() const
    {
        return kpsoot_;
    }
};

}

#endif
