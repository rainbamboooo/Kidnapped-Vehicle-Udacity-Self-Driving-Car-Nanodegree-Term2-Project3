/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <tuple>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 100;
	particles.resize(num_particles);
	weights.resize(num_particles);
	default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	for (int i = 0; i < num_particles; i++){
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;
		particles[i] = p;
	}
	is_initialized = true;
}


void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;
	for (int i = 0; i < num_particles; i++){
		double x, y, theta;
		if (fabs(yaw_rate) < 0.00001){
			x = particles[i].x + velocity * delta_t * cos(particles[i].theta);
			y = particles[i].y + velocity * delta_t * sin(particles[i].theta);
			theta = particles[i].theta;
		}
		else{
			x = particles[i].x + velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
			y = particles[i].y + velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
			theta = yaw_rate * delta_t + particles[i].theta;
		}
		normal_distribution<double> n_dist_x(x, std_pos[0]);
		normal_distribution<double> n_dist_y(y, std_pos[1]);
		normal_distribution<double> n_dist_theta(theta, std_pos[2]);
		particles[i].x = n_dist_x(gen);
		particles[i].y = n_dist_y(gen);
		particles[i].theta = n_dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (int i = 0; i < observations.size(); i++){
		double min_dis = 10000000;
		for (int j = 0; j < predicted.size(); j++){
			double dis = dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);
			if (dis < min_dis){
				min_dis = dis;
				observations[i].id = j;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
  
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
  //   http://planning.cs.uiuc.edu/node99.html
	for (int i=0; i < num_particles; i++){
		Particle p = particles[i];

		std::vector<LandmarkObs> predicted_landmarks;
		for (auto lm : map_landmarks.landmark_list){
			LandmarkObs single_lm;
			single_lm.x = lm.x_f;
			single_lm.y = lm.y_f;
			single_lm.id = lm.id_i;
			double dx = single_lm.x - p.x;
			double dy = single_lm.y - p.y;

			if (dx*dx + dy*dy <= sensor_range*sensor_range)
				predicted_landmarks.push_back(single_lm);
		}

		std::vector<LandmarkObs> transformed_obs;
    		for (auto obs : observations){
			LandmarkObs single_t_obs;
			single_t_obs.x = p.x + obs.x * cos(p.theta) - obs.y * sin(p.theta);
			single_t_obs.y = p.y + obs.x * sin(p.theta) + obs.y * cos(p.theta);
			single_t_obs.id = obs.id;
			transformed_obs.push_back(single_t_obs);
		}

		dataAssociation(predicted_landmarks, transformed_obs);
		
		double prob = 1.0f;
		for (int j=0; j < transformed_obs.size(); j++){
			double obs = transformed_obs[j];
			double assoc_lm = predicted_landmarks[obs.id];
			double x_diff = obs.x - assoc_lm.x;
			double y_diff = obs.y - assoc_lm.y;
			double n_x = 2 * std_landmark[0] * std_landmark[0];
			double n_y = 2 * std_landmark[1] * std_landmark[1];

			double pdf = 1 / (2 * std_landmark[0] * std_landmark[1] * M_PI) * exp(-(x_diff * x_diff / n_x + y_diff * y_diff / n_y));

			prob *= pdf;
		}
	particles[i].weight = prob;
	weights[i] = prob;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  default_random_engine gen;
  std::discrete_distribution<int> d(weights.begin(), weights.end());
  std::vector<Particle> new_particles;
  for(unsigned i = 0; i < num_particles; i++)
  {
    auto ind = d(gen);
    new_particles.push_back(std::move(particles[ind]));
  }
  particles = std::move(new_particles);
}


Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
