#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <stdio.h>
#include <iostream>
#include "object.h"
#include "capture_image.h"
#include "algorithm.h"

#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_blas.h>


// general vars
int i = 0, j = 0, k = 0, l_ = 0; // indices
int pt_count = 0, row_counter = 0; // counters
double accum1 = 0.0, accum2 = 0.0, ack1 = 0.0, ack2 = 0.0, ach1 = 0.0, ach2 = 0.0; // accums
vector<double> rho(N);
vector<Point2f> c_est(N), c_approx(N), c_des(N);
double arc_length = 0.0;
const int N_harmonics = 5, L_samples = 50, T_meas = 5, D_samples = 8;
const int P_param = 4*N_harmonics+2;
double G_vec[2*L_samples][4*N_harmonics+2] = {{0.0}}; // Fourier long matrix
double G_target[2*L_samples][4*N_harmonics+2] = {{0.0}}; // Fourier long matrix
vector<double> c_vec(2*L_samples); // countour long vector
vector<double> s(4*N_harmonics+2); // shape parameters
// for pseudo-inverse
gsl_matrix *G_ = gsl_matrix_alloc(2*L_samples, 4*N_harmonics+2); // to store the data
gsl_matrix *cov_ = gsl_matrix_alloc(4*N_harmonics+2, 4*N_harmonics+2); // convariance matrix
gsl_multifit_linear_workspace * work_ = gsl_multifit_linear_alloc(2*L_samples,4*N_harmonics+2);
gsl_vector *c_ = gsl_vector_alloc(2*L_samples); // observations
gsl_vector *s_ = gsl_vector_alloc(4*N_harmonics+2); // shape parameters
double chisq; // some variable 
// for saving data
char *data = (char *)malloc(1000); // array to be used for storing data
FILE *pFile = NULL; // pointer to the file where the data is to be saved
bool DATA_FILE = false; // flag to control openning of the data file
bool G_TARGET_OK = false; // flag to control the display of target shape
double time_ = 0.0; // time variable
// for model approximation
Point2f robot_ref, delta_r, old_delta_r = Point2f(0,0); // ref, relative & old displacements
vector<double> s_ref(P_param), delta_s(P_param), s_approx(P_param); // shape param vecs
double delta_s_hat[P_param] = {0.0}; // estimated shape parm
bool REFERENCE = false, SET_DATA = false; // flags
double R[T_meas][2] = {{0.0}}; // for robot sampling positions matrices
double R_coll[T_meas][2] = {{0.0}}; // for robot sampling positions matrices
double Sigma[T_meas][P_param] = {{0.0}}; // for sampling shape param vector Σ_i
double Sigma_hat[T_meas][P_param] = {{0.0}}; // for sampling shape param vector Σ_i
double Sigma_coll[T_meas][P_param] = {{0.0}}; // for sampling shape param vector Σ_i
double E[T_meas][P_param] = {{0.0}}; // estimation error
double invR[2][T_meas] = {{0.0}}, det_ = 0.0; // (pseudo-)inverse position matrix
double Q[P_param][2] = {{0.0}}; // the estimated shape parameters
Point2f locality = Point2f(0,0); // to plot initial sampling point
// for control
vector<double> s_error(P_param); // error of shape parameters
double mag_s = 100; // magnitude of deformation parameters
//const double s_d[22] = {114.548, -9.93682, 22.3967, 94.3998, 3.43359, -0.46109, 0.71477, -4.3244, 5.73633, 4.31714, 4.08393, 0.00377737, 2.90455, -0.500866, 1.70678, 1.80662, 3.02008, 3.10568, 0.201307, 2.4813, 138.586, 280.603};
//const double s_d[22] = {193.442,19.7688,29.5944,71.8701,2.69481,-1.65607,-0.328007,-2.31682,20.1452,13.5343,10.9112,-7.52938,2.11358,-0.292037,2.48422,0.326107,8.94018,7.10183,-1.82351,5.13956,249.429,287.943};
//const double s_d[22] = {127.691,47.1268,-50.6058,73.2467,5.59785,-5.68514,-4.06557,-18.9982,8.34006,6.28275,-3.16754,-5.1164,6.65911,0.556981,-7.52153,-0.342836,0.958851,4.39931,-2.65744,-0.351727,173.408,233.871};
//const double s_d[22] = {82.9044,-16.9383,17.3171,101.586,-6.6825,-2.76072,-2.09452,2.81249,-8.7089,-0.361736,-1.44137,0.823815,-4.05852,-2.18935,-0.351124,-1.17677,-4.12801,-0.868129,-0.686709,-0.771583,101.895,267.205}; // compress
//const double s_d[22] = {125.215,14.1171,-8.93248,90.1334,1.96551,0.207195,1.46926,-3.13161,9.8139,3.53851,2.07754,-3.41114,2.1178,0.782233,0.460294,1.088,5.29767,3.26941,-0.519991,1.84221,159.932,240.851}; // stretch 1
const double s_d[22] = {139.406,20.7275,-10.1345,84.1567,2.07044,-1.87701,-1.40273,-2.33218,12.5129,6.11657,3.74655,-5.81637,2.47439,-0.790192,0.703412,0.643505,6.01959,5.34527,-1.0097,2.02633,182.157,247.076}; // stretch 2
gsl_matrix *Q_ = gsl_matrix_alloc(P_param,2); // to store the data
gsl_matrix *Qcov_ = gsl_matrix_alloc(2,2); // convariance matrix
gsl_multifit_linear_workspace * Qwork_ = gsl_multifit_linear_alloc(P_param,2);
gsl_vector *ds_ = gsl_vector_alloc(P_param); // observations
gsl_vector *dr_ = gsl_vector_alloc(2); // computed controls

// local functions
void feedback_shape_parameters( void ); // compute shape parameters s
void approximate_model( void ); // to estimate (approximate) model

// the main algorithm
void compute_algorithm( void ) {

				// compute shape parameters
				feedback_shape_parameters( ); // this function calculates the "s" param. vector

				// evaluate least-squares result (just to graphically compare the parameters)
				accum1 = 0.0; accum2 = 0.0; ack1 = 0.0; ack2 = 0.0; l_ = 0.0; // initialise to zero
				ach1 = 0.0; ach2 = 0.0; // zero
				for (i = 0; i < N+1; i += int(N/L_samples)) { // do for L samples
								accum1 = 0.0; accum2 = 0.0; ack1 = 0.0; ack2 = 0.0; // reset to zero
								ach1 = 0.0; ach2 = 0.0; // reset to zero
								for ( j = 0; j < (int)(G_->size2); j++) { // do for all parameters s
												accum1 += G_vec[0+2*l_][j]*(s[j]-s_error[j]); // Mat-vec multiplication
												accum2 += G_vec[1+2*l_][j]*(s[j]-s_error[j]); // Mat-vec multiplication
												ack1 += G_vec[0+2*l_][j]*s_approx[j];
												ack2 += G_vec[1+2*l_][j]*s_approx[j];
												ach1 += G_target[0+2*l_][j]*s_d[j];
												ach2 += G_target[1+2*l_][j]*s_d[j];
								}
								c_est[l_] = Point2f(accum1,accum2); // approximate curve's point
								c_approx[l_] = Point2f(ack1,ack2); // approximate curve's point
								c_des[l_] = Point2f(ach1,ach2); // approximate curve's point
								l_++; // increment
				}

				// compute online estimator
				if (SET_) approximate_model(); // if press enter key
				else REFERENCE = false; // clear flag for initial (zero) reference

				// automatic robot controller
				for (i = 0; i < (int)(ds_->size); i++) s_error[i] = saturate(s[i]-s_d[i],3.0);
				if ( CONTROL ) {
								// copy Q matrix into Q_
								for (i = 0; i < (int)(Q_->size1); i++) 
												for (j = 0; j < (int)(Q_->size2); j++) 
																gsl_matrix_set(Q_, i, j, Q[i][j]);
								// put "obervation" vector s into ds_
								for (i = 0; i < (int)(ds_->size); i++)
												gsl_vector_set(ds_, i, s_error[i]);

								// compute least-squares approximation
								gsl_multifit_linear (Q_, ds_, dr_, Qcov_, &chisq, Qwork_);

								// velocity control
								const double K_gain = 0.1;
								robot.x -= saturate(K_gain*gsl_vector_get(dr_,0),0.02);
								robot.y -= saturate(K_gain*gsl_vector_get(dr_,1),0.02);

				}
				// compute magnitude of shape error
				accum1 = 0.0;
				for (i = 0; i < (int)(ds_->size); i++) {
								accum1 += (s[i]-s_d[i])*(s[i]-s_d[i]);
								printf("%g,",s[i]); 
				}  //printf("\n");
				mag_s = sqrt(accum1);
				printf("\ncount = %d, t = %g, ||δs|| = %g\n",abs(pt_count-T_meas),time_,mag_s);

				// save data
				if (CONTROL) {
								if (DATA_FILE == 0) {  // create and open file
												pFile = fopen("saved_data.txt","wa"); // open data file
												DATA_FILE = 1; // block call
								}
								sprintf(data,"%g %g\n", time_, mag_s);
								fputs(data, pFile);
								time_ += 0.01;
				}


} // end of routine


// function to compute shape parameters s
void feedback_shape_parameters( void ) { 
				// compute arc ρ parameter
				arc_length = 0.0; // starts with 0
				rho[0] = 0.0; // no lenght at 0 displacement
				for ( i = 1; i < N; i++ ) { // do from 2nd until the end
								Point2f vec = c[i] - c[i-1]; // compute the vector joining two points
								arc_length += norm(vec); // compute its incremental length
								rho[i] = arc_length; // assign it to arc parameter
				}
				// normalise lenght parameter to 2π
				for (i = 0; i < N; i++ ) rho[i] = rho[i]/arc_length*2*PI;

				// construct feedback matrices (long regression matrix G_vec and vector c_vec)
				l_ = 0; // row counter
				for (i = 0; i < N; i += int(N/L_samples) ) { // L samples ρ_i along the curve 
								for (k = 0; k < N_harmonics; k++ ) { // to approx c with N harmonics
												// compute long regression-like matrix G(ρ_i)
												G_vec[0+2*l_][4*k+0] = cos((k+1)*rho[i]);
												G_vec[0+2*l_][4*k+1] = sin((k+1)*rho[i]); 
												G_vec[0+2*l_][4*k+2] = 0; 
												G_vec[0+2*l_][4*k+3] = 0; 
												G_vec[1+2*l_][4*k+0] = 0; 
												G_vec[1+2*l_][4*k+1] = 0; 
												G_vec[1+2*l_][4*k+2] = cos((k+1)*rho[i]);
												G_vec[1+2*l_][4*k+3] = sin((k+1)*rho[i]);
												// compute long curve vector c(ρ_i)
												c_vec[0+2*l_] = c[i].x;
												c_vec[1+2*l_] = c[i].y;
								}
								// append identity matrix to the end of G
								G_vec[0+2*l_][4*N_harmonics+0] = 1.0; G_vec[0+2*l_][4*N_harmonics+1] = 0.0;
								G_vec[1+2*l_][4*N_harmonics+0] = 0.0; G_vec[1+2*l_][4*N_harmonics+1] = 1.0;
								l_++; // increment
				}
				// put regression matrix data into G_
				for (i = 0; i < (int)(G_->size1); i++) { 
								for (j = 0; j < (int)(G_->size2); j++) {
												gsl_matrix_set(G_, i, j, G_vec[i][j]);
								}
				}

				// to display desired target as static
				if (!G_TARGET_OK) {
								for (i = 0; i < (int)(G_->size1); i++) {
												for (j = 0; j < (int)(G_->size2); j++) {
																G_target[i][j] = G_vec[i][j];
												}
								}
								G_TARGET_OK = true; // block call
				}

				// put "obervation" vector c(ρ) into c_
				for (i = 0; i < (int)(c_->size); i++)
								gsl_vector_set(c_, i, c_vec[i]);

				// compute least-squares approximation
				gsl_multifit_linear (G_, c_, s_, cov_, &chisq, work_);

				// put computed shape parameters into c
				for (i = 0; i < (int)(s_->size); i++) 
								s[i] = gsl_vector_get(s_,i);
}


// online estimator of the robot-object-shape model
void approximate_model() {
				// set references... do this only one time
				if (!REFERENCE) { // if press enter
								printf("Get sampling data\n");
								robot_ref = robot; s_ref = s; // get refs
								REFERENCE = true; // get refs only once; block call
				}

				// compute relative position and parameters
				delta_r = robot - robot_ref; // relative feedback position
				for (i = 0; i < (int)s.size(); i++) 
								delta_s[i] = s[i] - s_ref[i]; // relative feedback shape parameters

				// collect sampling data (robot positions and shape parameters)
				Point2f dist = delta_r - old_delta_r; // vector from old (last) position to current
				if ( norm(dist)> D_samples && fabs(dist.x)>1 && fabs(dist.y)>1 && pt_count<T_meas ) {
								// fill R matrix with robot's positions
								R_coll[pt_count][0] = delta_r.x; 
								R_coll[pt_count][1] = delta_r.y;
								cout << "R_coll = ["<<R_coll[pt_count][0]<<"]["<<R_coll[pt_count][1]<<"]\n";
								
								// fill Σ matrix with feedback shape parameters
								for (i = 0; i < (int)s.size(); i++) {
												Sigma_coll[pt_count][i] = delta_s[i];
												printf("Sample %dth paramter = %g\n",i,Sigma_coll[pt_count][i]); 
								} cout << endl;

								// end of k-th data collection
								if (++pt_count == T_meas) printf("Data Ready\n"); // increment counter
								old_delta_r = delta_r; // assign old measurement
				}

				// set the collected position data into the R matrix
				if (pt_count == T_meas) {
								for (k = 0; k < T_meas; k++) {
												R[k][0] = R_coll[k][0];
												R[k][1] = R_coll[k][1];
												for (i = 0; i < (int)s.size(); i++) { 
																Sigma[k][i] = Sigma_coll[k][i];
												}
								}
								if (!SET_DATA) SET_DATA = true; // we have one at least set of data
				}

				// evaluate model approximation
				if (SET_DATA) {
								// approximate deformation local model for s_i
								for (i = 0; i < (int)s.size(); i++) {
												// compute the online (current) term
												delta_s_hat[i] = delta_r.x*Q[i][0] + delta_r.y*Q[i][1];

												// compute estimated feedback vector
												for (k = 0; k < T_meas; k++) 
																Sigma_hat[k][i] = R[k][0]*Q[i][0] + R[k][1]*Q[i][1];

												// compute estimation error
												for (k = 0; k < T_meas; k++) 
																E[k][i] = Sigma_hat[k][i] - Sigma[k][i];

												// compute update rule
												const double gamma = 0.000005;
												accum1 = 0.0; accum2 = 0.0; // reset to zero
												for ( k = 0; k < T_meas; k++) { // for all samples
																accum1 += R[k][0]*E[k][i]; // multiplication
																accum2 += R[k][1]*E[k][i]; // multiplication
												}
												Q[i][0]-=gamma*(accum1 + delta_r.x*(delta_s_hat[i] - delta_s[i]));
												Q[i][1]-=gamma*(accum2 + delta_r.y*(delta_s_hat[i] - delta_s[i]));

												// compute estimated feedback shape parameters
												s_approx[i] = Q[i][0]*delta_r.x + Q[i][1]*delta_r.y + s_ref[i];
												//printf("%d-th param, %g, %g\n",i,s[i],s_approx[i]);
								}
								//printf("\n");
				} // */

				// collect new data set if outside the local region
				if (SET_DATA && pt_count == T_meas) {
								int mid = (int)T_meas/2;
								dist = Point2f(R[mid][0],R[mid][1]) - delta_r; // vec from δr(t0)->δr(t)
								if (norm(dist) > T_meas*D_samples) pt_count = 0; // reset counter
								locality = dist + delta_r + robot_ref; // set initial sampling point
				}

} // end of routine

// saturation
double saturate(double signal, double limit)
{
				double retval = 0.0; // to return value
				if (signal > limit) retval = limit; // if greater
				else if (signal < -limit) retval = -limit; // if leaser
				else retval = signal; // else
				return retval; // return saturated value
}
