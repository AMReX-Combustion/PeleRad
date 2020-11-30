#include <boost/filesystem.hpp>
#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/test/unit_test.hpp>
#define BOOST_TEST_MODULE boostreaddatabase
#include <iostream>
#include <vector>

namespace but = boost::unit_test;
namespace bfs = boost::filesystem;
namespace bio = boost::iostreams;

BOOST_AUTO_TEST_CASE(boost_read_database, *but::tolerance(0.00001))
{
    bfs::path kplco2("../../data/kpDB/kpl_co2.dat");

    BOOST_TEST(bfs::exists(kplco2));

    bio::stream<bio::mapped_file_source> data(kplco2);

    std::vector<float> T;
    std::vector<float> kpco2;

    size_t i = 0;

    for (float T_temp, kp_temp; data >> T_temp >> kp_temp;)
    {
        // std::cout << "Reading from file:" << T << " " << kp << '\n';
        T.push_back(T_temp);

        kpco2.push_back(kp_temp);

        i++;
    }

    BOOST_TEST(T[15] == 600.);
    BOOST_TEST(kpco2[15] == 0.33684464);
    BOOST_TEST(T[107] == 2440.);
    BOOST_TEST(kpco2[107] == 0.07427838);

    data.close();
}

