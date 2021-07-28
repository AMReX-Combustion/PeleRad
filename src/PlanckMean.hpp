#ifndef PLANCK_MEAN_HPP
#define PLANCK_MEAN_HPP

#include <boost/filesystem.hpp>
#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/iostreams/stream.hpp>

#include <AMReX.H>
#include <AMReX_Gpu.H>

namespace PeleRad
{

class PlanckMean
{
private:
    using bfs_path = boost::filesystem::path;
    using bio_mappedsrc
        = boost::iostreams::stream<boost::iostreams::mapped_file_source>;

    amrex::GpuArray<amrex::Real, 126ul> kpco2_;
    amrex::GpuArray<amrex::Real, 126ul> kph2o_;
    amrex::GpuArray<amrex::Real, 126ul> kpco_;
    amrex::GpuArray<amrex::Real, 126ul> kpsoot_;

public:
    PlanckMean() = default;

    AMREX_GPU_HOST
    PlanckMean(std::string data_path)
    {
        bfs_path kplco2(data_path + "kpl_co2.dat");
        bfs_path kplh2o(data_path + "kpl_h2o.dat");
        bfs_path kplco(data_path + "kpl_co.dat");
        bfs_path kplsoot(data_path + "kpl_soot.dat");

        boost::filesystem::exists(kplco2);
        boost::filesystem::exists(kplh2o);
        boost::filesystem::exists(kplco);
        boost::filesystem::exists(kplsoot);

        bio_mappedsrc dataco2(kplco2);
        bio_mappedsrc datah2o(kplh2o);
        bio_mappedsrc dataco(kplco);
        bio_mappedsrc datasoot(kplsoot);

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
    }

    AMREX_GPU_HOST_DEVICE
    constexpr const amrex::GpuArray<amrex::Real, 126ul>& kpco2() const
    {
        return kpco2_;
    }

    AMREX_GPU_HOST_DEVICE
    constexpr const amrex::GpuArray<amrex::Real, 126ul>& kph2o() const
    {
        return kph2o_;
    }

    AMREX_GPU_HOST_DEVICE
    constexpr const amrex::GpuArray<amrex::Real, 126ul>& kpco() const
    {
        return kpco_;
    }

    AMREX_GPU_HOST_DEVICE
    constexpr const amrex::GpuArray<amrex::Real, 126ul>& kpsoot() const
    {
        return kpsoot_;
    }
};

}

#endif
