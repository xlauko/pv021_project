#pragma once

namespace filter {

using Image = cv::Mat;

namespace {
    Image get_map ( Image & img ) {
        Image res = img;
        //cv::threshold( img, res, 214, 255, cv::THRESH_BINARY );
        //cv::threshold( cv::Scalar::all( 255 ) - img, res, 214, 255, cv::THRESH_BINARY );
        //cv::blur( res, res, cv::Size( 3, 3 ) );
        /*cv::Canny( res, res, 100, 200, 3);

        cv::Mat element = cv::getStructuringElement( cv::MORPH_ELLIPSE, cv::Size( 5, 5 ) );

        cv::morphologyEx( res, res, cv::MORPH_CLOSE, element );*/
        return res;
    }
}

static Image filter ( Image & img ) {
	Image res;
    //auto map = get_map( img );
	//cv::Canny( img, res, 40, 200, 3 );
	//return res - map;
    cv::threshold( img, res, 200, 255, cv::THRESH_BINARY );
    return res;
}

} /* namespace filter */
