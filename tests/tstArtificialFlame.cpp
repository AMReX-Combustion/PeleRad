#include <boost/test/unit_test.hpp>

#define BOOST_TEST_MODULE amrexgetradprop

#include <PlanckMean.hpp>
#include <SpectralModels.hpp>

#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>

#include <iostream>
#include <vector>

AMREX_GPU_DEVICE AMREX_FORCE_INLINE void initGasField(int i, int j, int k,
    amrex::Array4<amrex::Real> const& y_co2,
    amrex::Array4<amrex::Real> const& y_h2o,
    amrex::Array4<amrex::Real> const& y_co,
    amrex::Array4<amrex::Real> const& fv_soot,
    amrex::Array4<amrex::Real> const& temp,
    amrex::Array4<amrex::Real> const& pressure,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& plo,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& phi)
{
    amrex::Real xc = (phi[0] + plo[0]) * 0.5;
    amrex::Real yc = (phi[1] + plo[1]) * 0.5;

    amrex::Real x = plo[0] + (i + 0.5) * dx[0];
    amrex::Real y = plo[1] + (j + 0.5) * dx[1];
    amrex::Real z = plo[2] + (k + 0.5) * dx[2];

    amrex::Real r = std::sqrt((x - xc) * (x - xc) + (y - yc) * (y - yc));

    amrex::Real expr  = std::exp(-(0.15 * r / (0.05 + 0.1 * 0.5 * z))
                                * (0.15 * r / (0.05 + 0.1 * 0.5 * z)));
    amrex::Real expTz = std::exp(-((0.275 * z - 1.3) / (0.7 + 0.5 * 0.5 * z))
                                 * ((0.275 * z - 1.3) / (0.7 + 0.5 * 0.5 * z)));

    temp(i, j, k) = 300.0 + 1800.0 * expr * expTz;

    pressure(i, j, k) = 1.0;

    amrex::Real expSoot
        = std::exp(-((0.0275 * z - 1.0) / 0.7) * ((0.0275 * z - 1.0) / 0.7));

    fv_soot(i, j, k) = 1e-6 * expr * expSoot;

    amrex::Real expCO2z
        = std::exp(-((0.275 * z - 1.1) / (0.6 + 0.5 * 0.5 * z))
                   * ((0.275 * z - 1.1) / (0.6 + 0.5 * 0.5 * z)));

    y_co2(i, j, k) = 0.09 * expr * expCO2z;

    amrex::Real expH2Oz
        = std::exp(-((0.275 * z - 1.0) / (0.7 + 0.5 * 0.5 * z))
                   * ((0.275 * z - 1.0) / (0.7 + 0.5 * 0.5 * z)));

    y_h2o(i, j, k) = 0.2 * expr * expH2Oz;

    amrex::Real expCOz
        = std::exp(-((0.025 * z - 1.0) / 0.7) * ((0.025 * z - 1.0) / 0.7));

    y_co(i, j, k) = 0.07 * expr * expCOz;
}

BOOST_AUTO_TEST_CASE(amrex_get_radprop)
{
    bool constexpr WRITE = true;

    std::string data_path;

#ifdef DATABASE_PATH
    data_path = DATABASE_PATH;
#else
    data_path = "../../data/kpDB/";
#endif

    using PeleRad::PlanckMean;
    using PeleRad::RadProp::getRadProp;

    PlanckMean radprop(data_path);

    amrex::Geometry geom;
    amrex::BoxArray grids;
    amrex::DistributionMapping dmap;

    amrex::MultiFab y_co2;
    amrex::MultiFab y_h2o;
    amrex::MultiFab y_co;
    amrex::MultiFab soot_fv_rad;
    amrex::MultiFab temperature;
    amrex::MultiFab pressure;
    amrex::MultiFab absc;

    std::cout << "initialize grid and variables ... \n";

    amrex::RealBox rb({ D_DECL(0.0, 0.0, 0.0) }, { D_DECL(0.125, 0.125, 0.8) });
    amrex::Array<int, AMREX_SPACEDIM> is_periodic { AMREX_D_DECL(0, 0, 0) };
    amrex::Geometry::Setup(&rb, 0, is_periodic.data());

    std::vector<int> npts { 128, 128, 128 };
    amrex::Box domain0(amrex::IntVect(AMREX_D_DECL(0, 0, 0)),
        amrex::IntVect { AMREX_D_DECL(npts[0] - 1, npts[1] - 1, npts[2] - 1) });
    grids.define(domain0);

    int const max_grid_size = 32;
    grids.maxSize(max_grid_size);

    amrex::IntVect ng = amrex::IntVect { 1 };

    dmap.define(grids);
    y_co2.define(grids, dmap, 1, 0);
    y_h2o.define(grids, dmap, 1, 0);
    y_co.define(grids, dmap, 1, 0);
    soot_fv_rad.define(grids, dmap, 1, 0);
    temperature.define(grids, dmap, 1, 0);
    pressure.define(grids, dmap, 1, 0);
    absc.define(grids, dmap, 1, 0);

    std::cout << "obtain the properties ... \n";
    int num_grow              = 0;
    int const num_rad_species = 1;

    const auto& kpco2  = radprop.kpco2();
    const auto& kph2o  = radprop.kph2o();
    const auto& kpco   = radprop.kpco();
    const auto& kpsoot = radprop.kpsoot();

    auto const plo = geom.ProbLoArray();
    auto const phi = geom.ProbHiArray();
    auto const dx  = geom.CellSizeArray();

    for (amrex::MFIter mfi(temperature, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi)
    {
        amrex::Box const& bx = mfi.validbox();

        auto const& Yco2  = y_co2.array(mfi);
        auto const& Yh2o  = y_h2o.array(mfi);
        auto const& Yco   = y_co.array(mfi);
        auto const& fv    = soot_fv_rad.array(mfi);
        auto const& T     = temperature.array(mfi);
        auto const& P     = pressure.array(mfi);
        auto const& kappa = absc.array(mfi);

        amrex::ParallelFor(
            bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                initGasField(i, j, k, Yco2, Yh2o, Yco, fv, T, P, dx, plo, phi);
            });

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            getRadProp(i, j, k, Yco2, Yh2o, Yco, fv, T, P, kappa, kpco2, kph2o,
                kpco, kpsoot);
        });
    }

    if (WRITE)
    {
        std::string pltfile("pltAF");

        amrex::MultiFab VarPlt(grids, dmap, 7, 0);
        amrex::MultiFab::Copy(VarPlt, y_co2, 0, 0, 1, 0);
        amrex::MultiFab::Copy(VarPlt, y_h2o, 0, 1, 1, 0);
        amrex::MultiFab::Copy(VarPlt, y_co, 0, 2, 1, 0);
        amrex::MultiFab::Copy(VarPlt, soot_fv_rad, 0, 3, 1, 0);
        amrex::MultiFab::Copy(VarPlt, temperature, 0, 4, 1, 0);
        amrex::MultiFab::Copy(VarPlt, pressure, 0, 5, 1, 0);
        amrex::MultiFab::Copy(VarPlt, absc, 0, 6, 1, 0);
        WriteSingleLevelPlotfile(pltfile, VarPlt,
            { "Y_CO2", "Y_H2O", "Y_CO", "fv", "T", "P", "Kappa" }, geom, 0.0,
            0);
    }

    BOOST_TEST(true);
}
