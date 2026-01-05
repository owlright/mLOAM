/*******************************************************
 *  Project: MiniLOAM
 *  Description:
 *    A minimal, self-contained LOAM-style demo:
 *      - Synthetic map point cloud (wall + ground)
 *      - Apply known SE(3) transform
 *      - KD-tree nearest neighbor association
 *      - PCA plane fitting
 *      - Point-to-plane residual
 *      - Ceres optimization with SE(3) local parameterization
 *      - PCL visualization
 *
 *  This file intentionally keeps everything in ONE cpp
 *  to expose the full geometric and optimization pipeline.
 *******************************************************/

 #include <iostream>
 #include <random>

 #include <Eigen/Core>
 #include <Eigen/Geometry>
 #include <Eigen/Eigenvalues>

 #include <ceres/ceres.h>
 #include <ceres/local_parameterization.h>

 #include <pcl/point_cloud.h>
 #include <pcl/point_types.h>
 #include <pcl/kdtree/kdtree_flann.h>
 #include <pcl/visualization/pcl_visualizer.h>

 using PointT = pcl::PointXYZ;
 using CloudT = pcl::PointCloud<PointT>;


 // =====================================================
 // Generate synthetic map cloud (ground + wall + noise)
 // =====================================================
 CloudT::Ptr GenerateMapCloud()
 {
     CloudT::Ptr cloud(new CloudT);

     std::default_random_engine eng;
     std::normal_distribution<double> noise(0.0, 0.03);

     // Ground: z ≈ 0
     for (int i = 0; i < 800; ++i) {
         cloud->push_back(PointT(
             (rand() % 100) / 10.0,
             (rand() % 100) / 10.0,
             noise(eng)));
     }

     // Wall: x ≈ 5
     for (int i = 0; i < 800; ++i) {
         cloud->push_back(PointT(
             5.0 + noise(eng),
             (rand() % 100) / 10.0,
             (rand() % 50) / 10.0));
     }

     return cloud;
 }


 // =====================================================
 // Apply SE(3) transform to a point cloud
 // =====================================================
 CloudT::Ptr TransformCloud(
     const CloudT::Ptr& cloud,
     const Eigen::Quaterniond& q,
     const Eigen::Vector3d& t)
 {
     CloudT::Ptr out(new CloudT);
     for (const auto& p : cloud->points) {
         Eigen::Vector3d pt(p.x, p.y, p.z);
         Eigen::Vector3d pt_t = q * pt + t;
         out->push_back(PointT(
             pt_t.x(), pt_t.y(), pt_t.z()));
     }
     return out;
 }


 // =====================================================
 // PCA plane fitting from neighbor points
 // =====================================================
 bool FitPlane(
     const std::vector<Eigen::Vector3d>& pts,
     Eigen::Vector3d& center,
     Eigen::Vector3d& normal)
 {
     if (pts.size() < 5) return false;

     center.setZero();
     for (const auto& p : pts)
         center += p;
     center /= pts.size();

     Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
     for (const auto& p : pts) {
         Eigen::Vector3d d = p - center;
         cov += d * d.transpose();
     }

     Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
     normal = es.eigenvectors().col(0); // smallest eigenvalue

     return true;
 }


 // =====================================================
 // Point-to-plane residual (LOAM surface constraint)
 // =====================================================
 struct PointPlaneResidual
 {
     PointPlaneResidual(
         const Eigen::Vector3d& curr,
         const Eigen::Vector3d& plane_pt,
         const Eigen::Vector3d& normal)
         : curr_(curr), plane_pt_(plane_pt), normal_(normal) {}

     template <typename T>
     bool operator()(const T* q, const T* t, T* residual) const
     {
         Eigen::Quaternion<T> quat(q[0], q[1], q[2], q[3]);
         Eigen::Matrix<T, 3, 1> trans(t[0], t[1], t[2]);

         Eigen::Matrix<T, 3, 1> p =
             quat * curr_.cast<T>() + trans;

         residual[0] =
             normal_.cast<T>().dot(p - plane_pt_.cast<T>());

         return true;
     }

     static ceres::CostFunction* Create(
         const Eigen::Vector3d& curr,
         const Eigen::Vector3d& plane_pt,
         const Eigen::Vector3d& normal)
     {
         return new ceres::AutoDiffCostFunction<
             PointPlaneResidual, 1, 4, 3>(
             new PointPlaneResidual(curr, plane_pt, normal));
     }

     Eigen::Vector3d curr_;
     Eigen::Vector3d plane_pt_;
     Eigen::Vector3d normal_;
 };


 // =====================================================
 // Main
 // =====================================================
 int main()
 {
     std::cout << "==== MiniLOAM demo ====" << std::endl;

     // -------------------------------------------------
     // 1. Generate map & scan
     // -------------------------------------------------
     CloudT::Ptr map_cloud = GenerateMapCloud();

     Eigen::Quaterniond q_gt =
         Eigen::AngleAxisd(0.25, Eigen::Vector3d::UnitZ());
     Eigen::Vector3d t_gt(0.6, 0.3, 0.2);

     CloudT::Ptr scan_cloud =
         TransformCloud(map_cloud, q_gt, t_gt);

     // -------------------------------------------------
     // 2. KD-tree on map
     // -------------------------------------------------
     pcl::KdTreeFLANN<PointT> kdtree;
     kdtree.setInputCloud(map_cloud);

     // -------------------------------------------------
     // 3. Ceres parameters (SE(3))
     // -------------------------------------------------
     double q[4] = {1.0, 0.0, 0.0, 0.0};
     double t[3] = {0.0, 0.0, 0.0};

     ceres::Problem problem;
     problem.AddParameterBlock(
         q, 4, new ceres::EigenQuaternionParameterization());
     problem.AddParameterBlock(t, 3);

     // -------------------------------------------------
     // 4. KD-tree association + plane fitting
     // -------------------------------------------------
     std::vector<int> indices(5);
     std::vector<float> distances(5);

     for (const auto& p : scan_cloud->points) {

         if (kdtree.nearestKSearch(
                 p, 5, indices, distances) < 5)
             continue;

         std::vector<Eigen::Vector3d> neighbors;
         for (int idx : indices) {
             const auto& mp = map_cloud->points[idx];
             neighbors.emplace_back(mp.x, mp.y, mp.z);
         }

         Eigen::Vector3d center, normal;
         if (!FitPlane(neighbors, center, normal))
             continue;

         Eigen::Vector3d curr(p.x, p.y, p.z);
         problem.AddResidualBlock(
             PointPlaneResidual::Create(curr, center, normal),
             nullptr, q, t);
     }

     // -------------------------------------------------
     // 5. Solve
     // -------------------------------------------------
     ceres::Solver::Options options;
     options.linear_solver_type = ceres::DENSE_QR;
     options.max_num_iterations = 20;
     options.minimizer_progress_to_stdout = true;

     ceres::Solver::Summary summary;
     ceres::Solve(options, &problem, &summary);

     std::cout << summary.BriefReport() << std::endl;

     Eigen::Quaterniond q_est(q[0], q[1], q[2], q[3]);
     Eigen::Vector3d t_est(t[0], t[1], t[2]);

     std::cout << "\nGround truth translation : "
               << t_gt.transpose()
               << "\nEstimated translation    : "
               << t_est.transpose()
               << std::endl;

     // -------------------------------------------------
     // 6. Visualization
     // -------------------------------------------------
     CloudT::Ptr aligned_cloud =
         TransformCloud(scan_cloud, q_est, t_est);

     pcl::visualization::PCLVisualizer viewer("MiniLOAM");
     viewer.addPointCloud(map_cloud, "map");
     viewer.addPointCloud(scan_cloud, "scan");
     viewer.addPointCloud(aligned_cloud, "aligned");

     viewer.setPointCloudRenderingProperties(
         pcl::visualization::PCL_VISUALIZER_COLOR,
         0.0, 1.0, 0.0, "map");
     viewer.setPointCloudRenderingProperties(
         pcl::visualization::PCL_VISUALIZER_COLOR,
         1.0, 0.0, 0.0, "scan");
     viewer.setPointCloudRenderingProperties(
         pcl::visualization::PCL_VISUALIZER_COLOR,
         0.0, 0.0, 1.0, "aligned");

     while (!viewer.wasStopped()) {
         viewer.spinOnce(50);
     }

     return 0;
 }
