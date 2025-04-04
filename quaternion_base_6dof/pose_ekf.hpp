#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <vector>
#include <iostream>
#include <cmath>

/// @brief Quaternion-based Extended Kalman Filter (EKF) for pose estimation
class PoseEKF
{
private:
    // State: [position(3), velocity(3), quaternion(4), angular_velocity(3)]
    Eigen::Vector<double, 13> x_;     // State vector (13x1)
    Eigen::Matrix<double, 13, 13> P_; // Covariance matrix (13x13)
    Eigen::Matrix<double, 13, 13> Q_; // Process noise covariance
    Eigen::Matrix<double, 7, 7> R_;   // Measurement noise covariance
    double dt_;                       // Time step (1.0s for 1Hz)
    const double kEpsilon_ = 1e-6;    // Threshold for small angle detection

    // Helper function to normalize quaternion in the state vector
    void normalizeQuaternion()
    {
        Eigen::Quaterniond q(x_(6), x_(7), x_(8), x_(9));
        q.normalize();
        x_(6) = q.w();
        x_(7) = q.x();
        x_(8) = q.y();
        x_(9) = q.z();
    }

    // Convert angular velocity to quaternion derivative
    // Only used for computing Jacobians
    Eigen::Vector4d angularVelToQuaternionDot(const Eigen::Vector4d &q, const Eigen::Vector3d &omega) const
    {
        // q = [w, x, y, z]
        // Compute quaternion derivative from angular velocity
        Eigen::Vector4d q_dot;
        q_dot(0) = -0.5 * (q(1) * omega(0) + q(2) * omega(1) + q(3) * omega(2));
        q_dot(1) = 0.5 * (q(0) * omega(0) + q(2) * omega(2) - q(3) * omega(1));
        q_dot(2) = 0.5 * (q(0) * omega(1) + q(3) * omega(0) - q(1) * omega(2));
        q_dot(3) = 0.5 * (q(0) * omega(2) + q(1) * omega(1) - q(2) * omega(0));
        return q_dot;
    }

    // Compute Jacobian of quaternion derivative with respect to quaternion
    Eigen::Matrix4d quaternionDotJacobian(const Eigen::Vector3d &omega) const
    {
        Eigen::Matrix4d J = Eigen::Matrix4d::Zero();

        J(0, 1) = -0.5 * omega(0);
        J(0, 2) = -0.5 * omega(1);
        J(0, 3) = -0.5 * omega(2);

        J(1, 0) = 0.5 * omega(0);
        J(1, 2) = 0.5 * omega(2);
        J(1, 3) = -0.5 * omega(1);

        J(2, 0) = 0.5 * omega(1);
        J(2, 1) = -0.5 * omega(2);
        J(2, 3) = 0.5 * omega(0);

        J(3, 0) = 0.5 * omega(2);
        J(3, 1) = 0.5 * omega(1);
        J(3, 2) = -0.5 * omega(0);

        return J;
    }

    // Compute Jacobian of quaternion derivative with respect to angular velocity
    Eigen::Matrix<double, 4, 3> quaternionDotOmegaJacobian(const Eigen::Vector4d &q) const
    {
        Eigen::Matrix<double, 4, 3> J(4, 3);

        J(0, 0) = -0.5 * q(1);
        J(0, 1) = -0.5 * q(2);
        J(0, 2) = -0.5 * q(3);

        J(1, 0) = 0.5 * q(0);
        J(1, 1) = -0.5 * q(3);
        J(1, 2) = 0.5 * q(2);

        J(2, 0) = 0.5 * q(3);
        J(2, 1) = 0.5 * q(0);
        J(2, 2) = -0.5 * q(1);

        J(3, 0) = -0.5 * q(2);
        J(3, 1) = 0.5 * q(1);
        J(3, 2) = 0.5 * q(0);

        return J;
    }

    // Compute state transition Jacobian for the full state vector
    Eigen::Matrix<double, 13, 13> computeStateTransitionJacobian(const Eigen::Vector3d &angular_velocity,
                                                                 const Eigen::Vector4d &quaternion) const
    {
        Eigen::Matrix<double, 13, 13> F = Eigen::Matrix<double, 13, 13>::Identity();

        // Position derivative wrt velocity
        F.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity() * dt_;

        // For 1Hz sampling with potentially large rotations,
        // we need a more accurate Jacobian for quaternion update

        const double angle = angular_velocity.norm() * dt_;

        if (angle > kEpsilon_)
        {
            // For significant rotation, use the exponential map Jacobian
            const Eigen::Vector3d axis = angular_velocity.normalized();
            const double half_angle = 0.5 * angle;
            const double sin_half_angle = std::sin(half_angle);
            const double cos_half_angle = std::cos(half_angle);

            // Quaternion update Jacobian
            Eigen::Matrix4d Q_wrt_q = Eigen::Matrix4d::Identity();
            Q_wrt_q(0, 0) = cos_half_angle;
            Q_wrt_q(0, 1) = -sin_half_angle * axis(0);
            Q_wrt_q(0, 2) = -sin_half_angle * axis(1);
            Q_wrt_q(0, 3) = -sin_half_angle * axis(2);

            Q_wrt_q(1, 0) = sin_half_angle * axis(0);
            Q_wrt_q(1, 1) = cos_half_angle;
            Q_wrt_q(1, 2) = -sin_half_angle * axis(2);
            Q_wrt_q(1, 3) = sin_half_angle * axis(1);

            Q_wrt_q(2, 0) = sin_half_angle * axis(1);
            Q_wrt_q(2, 1) = sin_half_angle * axis(2);
            Q_wrt_q(2, 2) = cos_half_angle;
            Q_wrt_q(2, 3) = -sin_half_angle * axis(0);

            Q_wrt_q(3, 0) = sin_half_angle * axis(2);
            Q_wrt_q(3, 1) = -sin_half_angle * axis(1);
            Q_wrt_q(3, 2) = sin_half_angle * axis(0);
            Q_wrt_q(3, 3) = cos_half_angle;

            F.block<4, 4>(6, 6) = Q_wrt_q;

            // Quaternion derivative wrt angular velocity
            Eigen::Matrix<double, 4, 3> Q_wrt_omega;
            const double dt_half = dt_ * 0.5;
            const double c1 = sin_half_angle / angular_velocity.norm();
            const double c2 = (1.0 - cos_half_angle) / (angular_velocity.squaredNorm());

            // Derivative of quaternion wrt angular velocity components
            for (int i = 0; i < 3; ++i)
            {
                Q_wrt_omega(0, i) = -dt_half * quaternion(i + 1) + c1 * axis(i) * quaternion(0) - c2 * angular_velocity(i) * quaternion(0);

                for (int j = 1; j < 4; ++j)
                {
                    if (i + 1 == j)
                    {
                        Q_wrt_omega(j, i) = dt_half * quaternion(0);
                    }
                    else
                    {
                        Q_wrt_omega(j, i) = 0.0;
                    }

                    Q_wrt_omega(j, i) += c1 * axis(i) * quaternion(j) - c2 * angular_velocity(i) * quaternion(j);
                }
            }

            F.block<4, 3>(6, 10) = Q_wrt_omega;
        }
        else
        {
            // For very small rotations, we can use the first-order approximation
            F.block<4, 4>(6, 6) = Eigen::Matrix4d::Identity() + quaternionDotJacobian(angular_velocity) * dt_;
            F.block<4, 3>(6, 10) = quaternionDotOmegaJacobian(quaternion) * dt_;
        }

        return F;
    }

    // Split the prediction step into multiple smaller steps for better accuracy
    void detailedPredict(const int num_steps)
    {
        const double small_dt = dt_ / num_steps;
        const double original_dt = dt_;

        // Temporarily set dt_ to the smaller step size
        dt_ = small_dt;

        // Perform multiple small prediction steps
        for (int i = 0; i < num_steps; ++i)
        {
            predict(false); // Don't update covariance until the final step
        }

        // Restore the original dt_
        dt_ = original_dt;

        // Update covariance once at the end using the full dt_
        const Eigen::Vector3d angular_velocity = x_.segment<3>(10);
        const Eigen::Vector4d quaternion(x_(6), x_(7), x_(8), x_(9));

        const Eigen::Matrix<double, 13, 13> F = computeStateTransitionJacobian(angular_velocity, quaternion);

        // Update covariance
        P_ = F * P_ * F.transpose() + Q_;
    }

public:
    // Constructor
    PoseEKF(const double dt = 0.1) : dt_(dt)
    {
        // Initialize state vector
        x_ = Eigen::Vector<double, 13>::Zero();
        x_(6) = 1.0; // w component of quaternion = 1 (identity rotation)

        // Initialize covariance matrices
        P_ = Eigen::Matrix<double, 13, 13>::Identity();

        // Process noise (adjusted for 1Hz sampling with potential large rotations)
        Q_ = Eigen::Matrix<double, 13, 13>::Identity();
        Q_.block<3, 3>(0, 0) *= 0.05;  // Position noise
        Q_.block<3, 3>(3, 3) *= 0.2;   // Velocity noise
        Q_.block<4, 4>(6, 6) *= 0.02;  // Quaternion noise
        Q_.block<3, 3>(10, 10) *= 0.3; // Angular velocity noise

        // Measurement noise (can be tuned based on your sensors)
        R_ = Eigen::Matrix<double, 7, 7>::Identity();
        // R_.block<3, 3>(0, 0) *= 0.1;        // Position measurement noise
        // R_.block<4, 4>(3, 3) *= 0.05;       // Quaternion measurement noise
        R_.block<3, 3>(0, 0) *= 0.5; // Position measurement noise
        R_.block<4, 4>(3, 3) *= 0.2; // Quaternion measurement noise
    }

    // Initialize with initial state
    void initialize(const Eigen::Vector3d &position, const Eigen::Quaterniond &orientation,
                    const Eigen::Vector3d &velocity = Eigen::Vector3d::Zero(),
                    const Eigen::Vector3d &angular_velocity = Eigen::Vector3d::Zero())
    {
        x_.segment<3>(0) = position;
        x_.segment<3>(3) = velocity;
        x_(6) = orientation.w();
        x_(7) = orientation.x();
        x_(8) = orientation.y();
        x_(9) = orientation.z();
        x_.segment<3>(10) = angular_velocity;

        // Make sure quaternion is normalized
        normalizeQuaternion();
    }

    // Set process noise covariance
    void setProcessNoise(const Eigen::Matrix<double, 13, 13> &Q)
    {
        Q_ = Q;
    }

    // Set measurement noise covariance
    void setMeasurementNoise(const Eigen::Matrix<double, 7, 7> &R)
    {
        R_ = R;
    }

    // Predict step with proper quaternion update
    void predict(const bool update_covariance = true)
    {
        // Extract current state components
        const Eigen::Vector3d position = x_.segment<3>(0);
        const Eigen::Vector3d velocity = x_.segment<3>(3);
        const Eigen::Quaterniond current_q(x_(6), x_(7), x_(8), x_(9)); // w, x, y, z order
        const Eigen::Vector3d angular_velocity = x_.segment<3>(10);

        // Predict position using velocity
        const Eigen::Vector3d new_position = position + velocity * dt_;

        // Predict quaternion using proper quaternion update
        const double angle = angular_velocity.norm() * dt_;
        Eigen::Quaterniond new_q;

        if (angle > kEpsilon_)
        {
            // For significant rotation, use quaternion multiplication
            const Eigen::Vector3d axis = angular_velocity.normalized();
            const Eigen::Quaterniond delta_q(Eigen::AngleAxisd(angle, axis));
            new_q = current_q * delta_q; // Right multiplication for local angular velocity
        }
        else
        {
            // For very small rotation, quaternion remains unchanged
            new_q = current_q;
        }

        // Update state vector
        x_.segment<3>(0) = new_position;
        x_(6) = new_q.w();
        x_(7) = new_q.x();
        x_(8) = new_q.y();
        x_(9) = new_q.z();

        // Normalize quaternion to ensure unit length
        normalizeQuaternion();

        // Update covariance matrix if requested
        if (update_covariance)
        {
            const Eigen::Vector4d quaternion(x_(6), x_(7), x_(8), x_(9));
            const Eigen::Matrix<double, 13, 13> F = computeStateTransitionJacobian(angular_velocity, quaternion);

            // Update covariance
            P_ = F * P_ * F.transpose() + Q_;
        }
    }

    // Advanced predict method that splits the prediction into multiple steps
    void advancedPredict(const int num_steps = 10)
    {
        if (num_steps <= 1)
        {
            predict(); // Use standard prediction
        }
        else
        {
            detailedPredict(num_steps); // Use detailed prediction with multiple steps
        }
    }

    // Update step using position and orientation measurements
    void update(const Eigen::Vector3d &measured_position, const Eigen::Quaterniond &measured_orientation)
    {
        // Normalize the quaternion measurement
        const Eigen::Quaterniond q_measured = measured_orientation.normalized();

        // Measurement vector [position(3), quaternion(4)]
        Eigen::Vector<double, 7> z;
        z.segment<3>(0) = measured_position;
        z(3) = q_measured.w();
        z(4) = q_measured.x();
        z(5) = q_measured.y();
        z(6) = q_measured.z();

        // Predicted measurement (directly from state)
        Eigen::Vector<double, 7> z_pred;
        z_pred.segment<3>(0) = x_.segment<3>(0); // position
        z_pred.segment<4>(3) = x_.segment<4>(6); // quaternion

        // Ensure we take the shortest path in quaternion space
        // (handling the dual-cover property of quaternions)
        const double dot_product = z_pred(3) * z(3) + z_pred(4) * z(4) +
                                   z_pred(5) * z(5) + z_pred(6) * z(6);
        if (dot_product < 0)
        {
            z(3) = -z(3);
            z(4) = -z(4);
            z(5) = -z(5);
            z(6) = -z(6);
        }

        // Measurement Jacobian
        Eigen::Matrix<double, 7, 13> H = Eigen::Matrix<double, 7, 13>::Zero();
        H.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity(); // position
        H.block<4, 4>(3, 6) = Eigen::Matrix4d::Identity(); // quaternion

        // Innovation (measurement residual)
        const Eigen::Vector<double, 7> y = z - z_pred;

        // Innovation covariance
        const Eigen::Matrix<double, 7, 7> S = H * P_ * H.transpose() + R_;

        // Kalman gain
        const Eigen::Matrix<double, 13, 7> K = P_ * H.transpose() * S.inverse();

        // Update state
        x_ = x_ + K * y;

        // Normalize quaternion again after update
        normalizeQuaternion();

        // Update covariance with Joseph form for numerical stability
        const Eigen::Matrix<double, 13, 13> I = Eigen::Matrix<double, 13, 13>::Identity();
        P_ = (I - K * H) * P_ * (I - K * H).transpose() + K * R_ * K.transpose();
    }

    // Get current filtered state
    void getState(Eigen::Vector3d &position, Eigen::Quaterniond &orientation,
                  Eigen::Vector3d &velocity, Eigen::Vector3d &angular_velocity) const
    {
        position = x_.segment<3>(0);
        velocity = x_.segment<3>(3);
        orientation = Eigen::Quaterniond(x_(6), x_(7), x_(8), x_(9)).normalized();
        angular_velocity = x_.segment<3>(10);
    }

    // Get state covariance
    Eigen::Matrix<double, 13, 13> getCovariance() const
    {
        return P_;
    }
};
