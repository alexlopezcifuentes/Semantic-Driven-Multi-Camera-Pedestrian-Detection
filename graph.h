#ifndef GRAPH_H
#define GRAPH_H

#include <iostream>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <list>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

class Graph
{
    using G = boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS>;
    using Component = int;
    using Mapping = std::map<G::vertex_descriptor, Component>;
    G g;

public:
    Graph(int V);
    void addEdge(int v, int w);
    void connectedComponents(cv::Mat &Connected, int NVertex);
};

#endif // GRAPH_H
