#include <boost/property_tree/ptree.hpp>
#include <boost/test/unit_test.hpp>
#define BOOST_TEST_MODULE boostpropertytree
#include <iostream>

namespace bpt = boost::property_tree;

bpt::ptree RadInputs() {
/*
  "RTE solver": {
    "P1": false,
    "Optically thin": true
  },
  "Spectral model": {
  "Constant": true,
  "Planck mean": false,
  "Gas Species": 3,
  "kappa": 1.9323
  }
*/
    bpt::ptree radsolver_tree;

    auto& rte_tree = radsolver_tree.add_child("RTE solver",bpt::ptree());
    rte_tree.put("P1",false);
    rte_tree.put("Optically thin",true);

    auto& spl_tree = radsolver_tree.add_child("Spectral model",bpt::ptree());
    spl_tree.put("Constant",true);
    spl_tree.put("Planck mean",false);
    spl_tree.put("Gas Species",3);
    spl_tree.put("kappa", 1.9323);

    return radsolver_tree;
}

BOOST_AUTO_TEST_CASE(boost_property_tree)
{
    auto radsolver = RadInputs();
    auto rte = radsolver.get_child("RTE solver");
    auto spl = radsolver.get_child("Spectral model");

    for (auto& op: radsolver)
    {
      std::cout<< op.first <<std::endl;
    }
    for (auto& op: rte)
    {
      std::cout<<op.first << ":" << op.second.data() <<std::endl;
    }
    for (auto& op: spl)
    {
      std::cout<<op.first << ":" << op.second.data() <<std::endl;
    }

    BOOST_TEST(rte.get<bool>("P1")==false);
    BOOST_TEST(!rte.get<bool>("Optically thin")==false);
    BOOST_TEST(!spl.get<bool>("Constant")==false);
    BOOST_TEST(spl.get<bool>("Planck mean")==false);
    BOOST_TEST(spl.get<int>("Gas Species")==3);
    BOOST_TEST(spl.get<double>("kappa")==1.9323);
}

