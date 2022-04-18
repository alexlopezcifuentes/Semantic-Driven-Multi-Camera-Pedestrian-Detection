#include "graph.h"
#include <iostream>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <list>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

Graph::Graph(int V) : g(V){
}

// Method to print connected components in an
// undirected graph
void Graph::connectedComponents(cv::Mat &Connected, int NVertex) {
    Mapping mappings;
    int n = boost::connected_components(g, boost::make_assoc_property_map(mappings));

    Connected = cv::Mat::zeros(n, NVertex, CV_8U);

    for (Component c = 0; c<n; ++c) {

        for (auto& mapping : mappings)
            if (mapping.second == c){
                Connected.at<uchar>(c,mapping.first) = 1;
            }
    }
}

void Graph::addEdge(int v, int w) {
    boost::add_edge(v, w, g);
}
