#include <boost/test/unit_test.hpp>

#define BOOST_TEST_MODULE amrexgetradprop

#include <PlanckMean.hpp>
#include <SpectralModels.hpp>

#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>

#include <iostream>
#include <vector>

using namespace amrex;

AMREX_GPU_HOST_DEVICE
inline void initGasField(int i, int j, int k, Array4<Real> const& yco2,
    Array4<Real> const& yh2o, Array4<Real> const& yco, const Array4<Real>& temp,
    const Array4<Real>& pressure, const GpuArray<Real, 3ul>& dx,
    const GpuArray<Real, 3ul>& plo, const GpuArray<Real, 3ul>& phi) noexcept
{
    constexpr int num_rad_species = 3;
    constexpr Real dTemp          = 1200;
    constexpr Real dPres          = 1.0;
    Real x                        = plo[0] + (i + 0.5) * dx[0];
    Real y                        = plo[1] + (j + 0.5) * dx[1];
    Real pi                       = 4.0 * std::atan(1.0);
    GpuArray<Real, 3> LL;
    GpuArray<Real, 3> PP;
    GpuArray<Real, 3> Y_lo;
    GpuArray<Real, 3> Y_hi;

    for (int n = 0; n < num_rad_species; n++)
    {
        LL[n] = phi[n] - plo[n];
        PP[n] = LL[n] / 2.0;
    }

    Y_lo[0] = 0.0;
    Y_hi[0] = 0.25;
    Y_lo[1] = 0.0;
    Y_hi[1] = 0.125;
    Y_lo[2] = 0.0;
    Y_hi[2] = 0.1;

    temp(i, j, k)     = 1500 + dTemp * std::sin(2.0 * pi * y / PP[1]);
    pressure(i, j, k) = 50.0 + dPres * std::cos(2.0 * pi * y / PP[1]);

    yco2(i, j, k) = Y_lo[0] + (Y_hi[0] - Y_lo[0]) * x / LL[0];
    yh2o(i, j, k) = Y_lo[1] + (Y_hi[1] - Y_lo[1]) * x / LL[1];
    yco(i, j, k)  = Y_lo[2] + (Y_hi[2] - Y_lo[2]) * x / LL[2];
}

BOOST_AUTO_TEST_CASE(amrex_get_radprop)
{
    constexpr bool WRITE = false;

    std::string data_path;

#ifdef DATABASE_PATH
    data_path = DATABASE_PATH;
#else
    data_path = "../../data/kpDB/";
#endif

    using PeleRad::PlanckMean;
    using PeleRad::RadProp::getRadPropGas;

    Vector<int> is_periodic(AMREX_SPACEDIM, 1);

    PlanckMean radprop(data_path);

    ParmParse pp;
    std::vector<int> npts { 128, 128, 128 };
    Box domain(IntVect(D_DECL(0, 0, 0)),
        IntVect(D_DECL(npts[0] - 1, npts[1] - 1, npts[2] - 1)));
    RealBox real_box(
        { D_DECL(0.0, 0.0, 0.0) }, { D_DECL(static_cast<double>(npts[0]) - 1.0,
                                       static_cast<double>(npts[1]) - 1.0,
                                       static_cast<double>(npts[2]) - 1.0) });
    Geometry geom;
    geom.define(domain, &real_box, CoordSys::cartesian, is_periodic.data());

    GpuArray<Real, 3ul> plo;
    GpuArray<Real, 3ul> phi;
    GpuArray<Real, 3ul> dx;

    for (int i = 0; i < AMREX_SPACEDIM; ++i)
    {
        phi[i] = domain.length(i);
        dx[i]  = (phi[i] - plo[i]) / domain.length(i);
    }

    int max_size = 32;
    pp.query("max_size", max_size);
    BoxArray ba(domain);
    ba.maxSize(max_size);

    ParmParse ppa("amr");

    DistributionMapping dm { ba };

    int num_grow = 0;

    MultiFab y_co2(ba, dm, 1, num_grow);
    MultiFab y_h2o(ba, dm, 1, num_grow);
    MultiFab y_co(ba, dm, 1, num_grow);
    MultiFab temperature(ba, dm, 1, num_grow);
    MultiFab pressure(ba, dm, 1, num_grow);

    const auto& kpco2 = radprop.kpco2();
    const auto& kph2o = radprop.kph2o();
    const auto& kpco  = radprop.kpco();

    MultiFab absc(ba, dm, 1, num_grow);

    for (MFIter mfi(temperature, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& gbox = mfi.tilebox();

        const Array4<Real>& Yco2  = y_co2.array(mfi);
        const Array4<Real>& Yh2o  = y_h2o.array(mfi);
        const Array4<Real>& Yco   = y_co.array(mfi);
        const Array4<Real>& T     = temperature.array(mfi);
        const Array4<Real>& P     = pressure.array(mfi);
        const Array4<Real>& kappa = absc.array(mfi);

        ParallelFor(gbox,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                initGasField(i, j, k, Yco2, Yh2o, Yco, T, P, dx, plo, phi);
                getRadPropGas(
                    i, j, k, Yco2, Yh2o, Yco, T, P, kappa, kpco2, kph2o, kpco);
            });
    }

    if (WRITE)
    {
        std::string pltfile("plt");
        ppa.query("plot_file", pltfile);

        MultiFab VarPlt(ba, dm, 6, num_grow);
        MultiFab::Copy(VarPlt, y_co2, 0, 0, 1, num_grow);
        MultiFab::Copy(VarPlt, y_h2o, 0, 1, 1, num_grow);
        MultiFab::Copy(VarPlt, y_co, 0, 2, 1, num_grow);
        MultiFab::Copy(VarPlt, temperature, 0, 3, 1, num_grow);
        MultiFab::Copy(VarPlt, pressure, 0, 4, 1, num_grow);
        MultiFab::Copy(VarPlt, absc, 0, 5, 1, num_grow);
        std::string initfile = amrex::Concatenate(pltfile, 99);
        WriteSingleLevelPlotfile(pltfile, VarPlt,
            { "Y_CO2", "Y_H2O", "Y_CO", "T", "P", "Kappa" }, geom, 0.0, 0);
    }

    BOOST_TEST(true);
}
