#ifndef PLANCK_MEAN_HPP
#define PLANCK_MEAN_HPP

#ifdef PELERAD_USE_HIP
#include <experimental/filesystem>
#else
#include <filesystem>
#endif
#include <fstream>

#include <AMReX.H>
#include <AMReX_Gpu.H>

namespace PeleRad {

class PlanckMean
{
private:
  amrex::GpuArray<amrex::Real, 126ul> kpco2_;
  amrex::GpuArray<amrex::Real, 126ul> kph2o_;
  amrex::GpuArray<amrex::Real, 126ul> kpco_;
  amrex::GpuArray<amrex::Real, 126ul> kpch4_;
  amrex::GpuArray<amrex::Real, 126ul> kpc2h4_;
  amrex::GpuArray<amrex::Real, 126ul> kpsoot_;

public:
  AMREX_GPU_HOST
  PlanckMean() = default;

  AMREX_GPU_HOST
  PlanckMean(std::string data_path) { load(data_path); }

  AMREX_GPU_HOST
  PlanckMean(PlanckMean const&) = delete;

  AMREX_GPU_HOST
  PlanckMean& operator=(PlanckMean const&) = delete;

  AMREX_GPU_HOST
  void load(std::string data_path)
  {
#ifdef PELERAD_USE_HIP
    using sfp = std::experimental::filesystem::path;
#else
    using sfp = std::filesystem::path;
#endif

    sfp kplco2(data_path + "kpl_co2.dat");
    sfp kplh2o(data_path + "kpl_h2o.dat");
    sfp kplco(data_path + "kpl_co.dat");
    sfp kplch4(data_path + "kpl_ch4.dat");
    sfp kplc2h4(data_path + "kpl_c2h4.dat");
    sfp kplsoot(data_path + "kpl_soot.dat");

    std::ifstream dataco2(kplco2);
    std::ifstream datah2o(kplh2o);
    std::ifstream dataco(kplco);
    std::ifstream datach4(kplch4);
    std::ifstream datac2h4(kplc2h4);
    std::ifstream datasoot(kplsoot);

    size_t i_co2 = 0;
    for (float T_temp, kp_temp; dataco2 >> T_temp >> kp_temp;) {
      kpco2_[i_co2] = kp_temp;
      i_co2++;
    }

    size_t i_h2o = 0;
    for (float T_temp, kp_temp; datah2o >> T_temp >> kp_temp;) {
      kph2o_[i_h2o] = kp_temp;
      i_h2o++;
    }

    size_t i_co = 0;
    for (float T_temp, kp_temp; dataco >> T_temp >> kp_temp;) {
      kpco_[i_co] = kp_temp;
      i_co++;
    }

    size_t i_ch4 = 0;
    for (float T_temp, kp_temp; datach4 >> T_temp >> kp_temp;) {
      kpch4_[i_ch4] = kp_temp;
      i_ch4++;
    }

    size_t i_c2h4 = 0;
    for (float T_temp, kp_temp; dataco >> T_temp >> kp_temp;) {
      kpc2h4_[i_c2h4] = kp_temp;
      i_c2h4++;
    }

    size_t i_soot = 0;
    for (float T_temp, kp_temp; datasoot >> T_temp >> kp_temp;) {
      kpsoot_[i_soot] = kp_temp;
      i_soot++;
    }

    dataco2.close();
    datah2o.close();
    dataco.close();
    datach4.close();
    datac2h4.close();
    datasoot.close();
  }

  AMREX_GPU_HOST_DEVICE
  const amrex::GpuArray<amrex::Real, 126ul>& kpco2() const { return kpco2_; }

  AMREX_GPU_HOST_DEVICE
  const amrex::GpuArray<amrex::Real, 126ul>& kph2o() const { return kph2o_; }

  AMREX_GPU_HOST_DEVICE
  const amrex::GpuArray<amrex::Real, 126ul>& kpco() const { return kpco_; }

  AMREX_GPU_HOST_DEVICE
  const amrex::GpuArray<amrex::Real, 126ul>& kpch4() const { return kpch4_; }

  AMREX_GPU_HOST_DEVICE
  const amrex::GpuArray<amrex::Real, 126ul>& kpc2h4() const { return kpc2h4_; }

  AMREX_GPU_HOST_DEVICE
  const amrex::GpuArray<amrex::Real, 126ul>& kpsoot() const { return kpsoot_; }
};

} // namespace PeleRad

#endif
