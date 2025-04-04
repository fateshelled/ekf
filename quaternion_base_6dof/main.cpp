#include <GL/glut.h>
#include <mutex>
#include <fstream>
#include <memory>
#include <iomanip>
#include <cstdlib>

#include "pose_ekf.hpp"

// 軌道データを保存するクラス
class TrajectoryData
{
public:
    struct PoseData
    {
        double timestamp;
        Eigen::Vector3d position;
        Eigen::Quaterniond orientation;
        bool is_filtered;

        PoseData(double time, const Eigen::Vector3d &pos, const Eigen::Quaterniond &ori, bool filtered = false)
            : timestamp(time), position(pos), orientation(ori), is_filtered(filtered) {}
    };

    std::vector<PoseData> raw_trajectory;      // 観測データの軌道
    std::vector<PoseData> filtered_trajectory; // フィルタリング済みの軌道
    std::vector<PoseData> ground_truth;        // 真の軌道
    std::mutex data_mutex;                     // スレッドセーフのためのミューテックス

    // データのクリア
    void clear()
    {
        std::lock_guard<std::mutex> lock(data_mutex);
        raw_trajectory.clear();
        filtered_trajectory.clear();
        ground_truth.clear();
    }

    void saveToFile(const std::string &filename) const
    {
        std::ofstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return;
        }

        file << "# Trajectory Data Export\n";
        file << "# timestamp,data_type,pos_x,pos_y,pos_z,quat_w,quat_x,quat_y,quat_z\n";
        // data_type: 0=raw, 1=filtered, 2=ground_truth

        // 生データの保存
        for (const auto &pose : raw_trajectory)
        {
            file << pose.timestamp << ",0,"
                 << pose.position.x() << "," << pose.position.y() << "," << pose.position.z() << ","
                 << pose.orientation.w() << "," << pose.orientation.x() << ","
                 << pose.orientation.y() << "," << pose.orientation.z() << "\n";
        }

        // フィルタリング済みデータの保存
        for (const auto &pose : filtered_trajectory)
        {
            file << pose.timestamp << ",1,"
                 << pose.position.x() << "," << pose.position.y() << "," << pose.position.z() << ","
                 << pose.orientation.w() << "," << pose.orientation.x() << ","
                 << pose.orientation.y() << "," << pose.orientation.z() << "\n";
        }

        // 真の軌道データの保存
        for (const auto &pose : ground_truth)
        {
            file << pose.timestamp << ",2,"
                 << pose.position.x() << "," << pose.position.y() << "," << pose.position.z() << ","
                 << pose.orientation.w() << "," << pose.orientation.x() << ","
                 << pose.orientation.y() << "," << pose.orientation.z() << "\n";
        }

        file.close();
        std::cout << "Saved trajectory data to: " << filename << std::endl;
    }
};

// 可視化を担当するクラス
class TrajectoryVisualizer
{
private:
    TrajectoryData *trajectory_data_;
    int selected_frame_index_;
    float camera_distance_;
    float camera_theta_;
    float camera_phi_;
    Eigen::Vector3d camera_target_;
    bool show_raw_;
    bool show_filtered_;
    bool show_coordinate_frames_;
    bool show_ground_plane_;
    bool show_ground_truth_;
    bool animation_running_;
    int animation_speed_;
    float axis_scale_;
    float robot_scale_;

    // 座標フレームを描画
    void drawCoordinateFrame(float scale = 1.0f) const
    {
        glLineWidth(2.0f);

        // X軸（赤）
        glBegin(GL_LINES);
        glColor3f(1.0f, 0.0f, 0.0f);
        glVertex3f(0.0f, 0.0f, 0.0f);
        glVertex3f(scale, 0.0f, 0.0f);
        glEnd();

        // Y軸（緑）
        glBegin(GL_LINES);
        glColor3f(0.0f, 1.0f, 0.0f);
        glVertex3f(0.0f, 0.0f, 0.0f);
        glVertex3f(0.0f, scale, 0.0f);
        glEnd();

        // Z軸（青）
        glBegin(GL_LINES);
        glColor3f(0.0f, 0.0f, 1.0f);
        glVertex3f(0.0f, 0.0f, 0.0f);
        glVertex3f(0.0f, 0.0f, scale);
        glEnd();

        glLineWidth(1.0f);
    }

    // グリッドを描画
    void drawGrid(int size, float step) const
    {
        glColor3f(0.7f, 0.7f, 0.7f);
        glLineWidth(1.0f);

        glBegin(GL_LINES);
        for (int i = -size; i <= size; i++)
        {
            // X方向の線
            glVertex3f(i * step, 0, -size * step);
            glVertex3f(i * step, 0, size * step);

            // Z方向の線
            glVertex3f(-size * step, 0, i * step);
            glVertex3f(size * step, 0, i * step);
        }
        glEnd();
    }

    // ロボットを描画（円柱と矢印で表現）
    void drawRobot(const Eigen::Vector3d &position, const Eigen::Quaterniond &orientation,
                   float size = 0.2f, bool is_filtered = false) const
    {
        glPushMatrix();

        // 位置に移動
        glTranslatef(position.x(), position.y(), position.z());

        // 回転を適用
        Eigen::AngleAxisd aa(orientation);
        glRotatef(aa.angle() * 180.0f / M_PI, aa.axis().x(), aa.axis().y(), aa.axis().z());

        // フィルタリング済みかどうかで色を変える
        if (is_filtered)
        {
            glColor3f(0.0f, 0.8f, 0.2f); // 緑色（フィルタリング済み）
        }
        else
        {
            glColor3f(1.0f, 0.4f, 0.4f); // 赤色（生データ）
        }

        // 本体（円柱）
        GLUquadricObj *quadric = gluNewQuadric();
        gluQuadricDrawStyle(quadric, GLU_FILL);
        gluCylinder(quadric, size * 0.8, size * 0.8, size * 1.2, 16, 1);

        // 上面
        glPushMatrix();
        glTranslatef(0, 0, size * 1.2);
        gluDisk(quadric, 0, size * 0.8, 16, 1);
        glPopMatrix();

        // 下面
        gluDisk(quadric, 0, size * 0.8, 16, 1);

        // 方向を示す矢印（X方向）
        glColor3f(1.0f, 0.0f, 0.0f);
        glPushMatrix();
        glTranslatef(0, 0, size * 0.6);
        glRotatef(90, 0, 1, 0);
        gluCylinder(quadric, size * 0.1, size * 0.1, size * 1.5, 8, 1);
        glTranslatef(0, 0, size * 1.5);
        glutSolidCone(size * 0.2, size * 0.4, 8, 1);
        glPopMatrix();

        gluDeleteQuadric(quadric);

        // 座標フレームを描画（オプション）
        if (show_coordinate_frames_)
        {
            drawCoordinateFrame(size * 2.0f);
        }

        glPopMatrix();
    }

    // 軌道を線で描画
    void drawTrajectory(const std::vector<TrajectoryData::PoseData> &trajectory,
                        int trajectory_type) const
    {
        if (trajectory.empty())
            return;

        // 軌道の種類によって色と線の太さを変える
        float color[3];
        switch (trajectory_type)
        {
        case 0: // 生データ（赤）
            glColor3f(1.0f, 0.4f, 0.4f);
            color[0] = 1.0f;
            color[1] = 0.4f;
            color[2] = 0.4f;
            glLineWidth(1.5f);
            break;
        case 1: // フィルタリング済み（緑）
            glColor3f(0.0f, 0.8f, 0.2f);
            color[0] = 0.0f;
            color[1] = 0.8f;
            color[2] = 0.2f;
            glLineWidth(3.0f);
            break;
        case 2: // 真の軌道（青）
            glColor3f(0.2f, 0.4f, 1.0f);
            color[0] = 0.2f;
            color[1] = 0.4f;
            color[2] = 1.0f;
            glLineWidth(2.0f);
            break;
        }

        // 軌道を線で描画
        glBegin(GL_LINE_STRIP);
        for (const auto &pose : trajectory)
        {
            glVertex3f(pose.position.x(), pose.position.y(), pose.position.z());
        }
        glEnd();

        glLineWidth(1.0f);

        // 10フレームごとに矢印を描画
        const int arrow_interval = 10; // 矢印を描画する間隔
        for (size_t i = 0; i < trajectory.size(); i += arrow_interval)
        {
            if (i < trajectory.size())
            {
                drawDirectionArrow(
                    trajectory[i].position,
                    trajectory[i].orientation,
                    0.7f, // 矢印のサイズ
                    color,
                    false // XYZ軸を表示するかどうか
                );
            }
        }

        // 選択されたフレームのロボットを表示
        if (!trajectory.empty() && trajectory_type != 2)
        { // GTはロボット表示しない
            int frame_index = selected_frame_index_;
            if (frame_index < 0)
                frame_index = 0;
            if (frame_index >= static_cast<int>(trajectory.size()))
            {
                frame_index = trajectory.size() - 1;
            }

            drawRobot(trajectory[frame_index].position,
                      trajectory[frame_index].orientation,
                      robot_scale_, trajectory_type == 1);
        }
        else if (!trajectory.empty() && trajectory_type == 2)
        {
            // GTの場合は小さい球体で表示
            int frame_index = selected_frame_index_;
            if (frame_index < 0)
                frame_index = 0;
            if (frame_index >= static_cast<int>(trajectory.size()))
            {
                frame_index = trajectory.size() - 1;
            }

            glPushMatrix();
            glTranslatef(
                trajectory[frame_index].position.x(),
                trajectory[frame_index].position.y(),
                trajectory[frame_index].position.z());
            glColor3f(0.2f, 0.4f, 1.0f);
            glutSolidSphere(robot_scale_ * 0.3, 12, 12);
            glPopMatrix();
        }
    }

    // クォータニオンをオイラー角（RPY: ロール・ピッチ・ヨー）に変換する関数
    Eigen::Vector3d quaternionToEuler(const Eigen::Quaterniond &q) const
    {
        // ZYXの回転順序でオイラー角を計算（ロール、ピッチ、ヨー）
        Eigen::Vector3d euler;

        // ロール (X軸周りの回転)
        euler(0) = atan2(2.0 * (q.w() * q.x() + q.y() * q.z()),
                         1.0 - 2.0 * (q.x() * q.x() + q.y() * q.y()));

        // ピッチ (Y軸周りの回転)
        // 特異点（ジンバルロック）近くでの数値的な問題を避けるためのクランプ
        double sinp = 2.0 * (q.w() * q.y() - q.z() * q.x());
        if (std::abs(sinp) >= 1.0)
            euler(1) = std::copysign(M_PI / 2.0, sinp); // 90度に制限
        else
            euler(1) = std::asin(sinp);

        // ヨー (Z軸周りの回転)
        euler(2) = atan2(2.0 * (q.w() * q.z() + q.x() * q.y()),
                         1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z()));

        // ラジアンから度に変換
        euler *= 180.0 / M_PI;

        return euler;
    }

    void drawDirectionArrow(const Eigen::Vector3d &position, const Eigen::Quaterniond &orientation,
                            float size, const float color[3], bool showXYZ = false) const
    {
        glPushMatrix();

        // 位置に移動
        glTranslatef(position.x(), position.y(), position.z());

        // 回転を適用
        Eigen::AngleAxisd aa(orientation);
        glRotatef(aa.angle() * 180.0f / M_PI, aa.axis().x(), aa.axis().y(), aa.axis().z());

        // 矢印の色を設定
        glColor3f(color[0], color[1], color[2]);

        // 固定された半径と長さで向きを表す矢印を描画
        GLUquadricObj *quadric = gluNewQuadric();
        gluQuadricDrawStyle(quadric, GLU_FILL);

        // X方向の矢印（前方向）
        glPushMatrix();
        glRotatef(90, 0, 1, 0);
        gluCylinder(quadric, size * 0.05, size * 0.05, size, 8, 1);
        glTranslatef(0, 0, size);
        glutSolidCone(size * 0.1, size * 0.3, 8, 1);
        glPopMatrix();

        // XYZ軸を表示するオプション
        if (showXYZ)
        {
            // Y方向の矢印（右方向）- 赤より薄い色
            glPushMatrix();
            glColor3f(color[0] * 0.7f, color[1] * 0.7f, color[2] * 0.7f);
            glRotatef(-90, 1, 0, 0);
            gluCylinder(quadric, size * 0.03, size * 0.03, size * 0.7, 8, 1);
            glTranslatef(0, 0, size * 0.7);
            glutSolidCone(size * 0.08, size * 0.2, 8, 1);
            glPopMatrix();

            // Z方向の矢印（上方向）- さらに薄い色
            glPushMatrix();
            glColor3f(color[0] * 0.5f, color[1] * 0.5f, color[2] * 0.5f);
            gluCylinder(quadric, size * 0.03, size * 0.03, size * 0.7, 8, 1);
            glTranslatef(0, 0, size * 0.7);
            glutSolidCone(size * 0.08, size * 0.2, 8, 1);
            glPopMatrix();
        }

        gluDeleteQuadric(quadric);
        glPopMatrix();
    }

    // 姿勢角を表示する関数
    void drawEulerAngles(const Eigen::Quaterniond &q_raw, const Eigen::Quaterniond &q_filtered,
                         const Eigen::Quaterniond &q_gt) const
    {
        // オイラー角に変換
        Eigen::Vector3d euler_raw = quaternionToEuler(q_raw);
        Eigen::Vector3d euler_filtered = quaternionToEuler(q_filtered);
        Eigen::Vector3d euler_gt = quaternionToEuler(q_gt);

        // 画面の上部に表示するテキスト
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2);
        oss << "Raw RPY: [" << euler_raw.x() << ", " << euler_raw.y() << ", " << euler_raw.z() << "] deg  |  ";
        oss << "Filtered RPY: [" << euler_filtered.x() << ", " << euler_filtered.y() << ", " << euler_filtered.z() << "] deg  |  ";
        oss << "GT RPY: [" << euler_gt.x() << ", " << euler_gt.y() << ", " << euler_gt.z() << "] deg";

        // 情報テキストとして描画
        drawInfoText(10, glutGet(GLUT_WINDOW_HEIGHT) - 40, oss.str());
    }

    // 情報テキストを描画
    void drawInfoText(int x, int y, const std::string &text) const
    {
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        gluOrtho2D(0, glutGet(GLUT_WINDOW_WIDTH), 0, glutGet(GLUT_WINDOW_HEIGHT));

        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();

        glColor3f(1.0f, 1.0f, 1.0f);
        glRasterPos2i(x, y);

        for (const char c : text)
        {
            glutBitmapCharacter(GLUT_BITMAP_9_BY_15, c);
        }

        glPopMatrix();
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
    }

    // カメラ位置を計算
    void updateCameraPosition()
    {
        const float x = camera_distance_ * sin(camera_theta_) * cos(camera_phi_);
        const float y = camera_distance_ * sin(camera_phi_);
        const float z = camera_distance_ * cos(camera_theta_) * cos(camera_phi_);

        gluLookAt(
            camera_target_.x() + x, camera_target_.y() + y, camera_target_.z() + z,
            camera_target_.x(), camera_target_.y(), camera_target_.z(),
            0.0, 1.0, 0.0);
    }

public:
    TrajectoryVisualizer(TrajectoryData *data)
        : trajectory_data_(data), selected_frame_index_(0), camera_distance_(10.0f),
          camera_theta_(M_PI / 4), camera_phi_(M_PI / 6), camera_target_(Eigen::Vector3d::Zero()),
          show_raw_(true), show_filtered_(true), show_coordinate_frames_(true),
          show_ground_plane_(true), animation_running_(false), animation_speed_(1),
          axis_scale_(1.0f), robot_scale_(0.5f) {}

    // 描画メイン関数
    void render()
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glLoadIdentity();

        // カメラ位置の設定
        updateCameraPosition();

        // グリッドを描画
        if (show_ground_plane_)
        {
            drawGrid(10, 1.0f);
        }

        // 世界座標系の原点に座標フレームを描画
        drawCoordinateFrame(axis_scale_);

        // 軌道データの描画
        // 真の軌道を最初に描画（他の軌道の下になるように）
        if (show_ground_truth_ && !trajectory_data_->ground_truth.empty())
        {
            drawTrajectory(trajectory_data_->ground_truth, 2);
        }

        // 生データを描画
        if (show_raw_ && !trajectory_data_->raw_trajectory.empty())
        {
            drawTrajectory(trajectory_data_->raw_trajectory, 0);
        }

        // フィルタリング済みデータを描画（最後に描画して前面に表示）
        if (show_filtered_ && !trajectory_data_->filtered_trajectory.empty())
        {
            drawTrajectory(trajectory_data_->filtered_trajectory, 1);
        }

        // 情報テキストの表示
        std::string info_text;
        if (!trajectory_data_->raw_trajectory.empty() &&
            selected_frame_index_ < static_cast<int>(trajectory_data_->raw_trajectory.size()))
        {
            std::ostringstream oss;
            oss << "Frame: " << selected_frame_index_;

            if (!trajectory_data_->filtered_trajectory.empty() &&
                selected_frame_index_ < static_cast<int>(trajectory_data_->filtered_trajectory.size()))
            {
                const auto &raw = trajectory_data_->raw_trajectory[selected_frame_index_];
                const auto &filtered = trajectory_data_->filtered_trajectory[selected_frame_index_];

                oss << " | Time: " << raw.timestamp
                    << " | Raw Pos: (" << raw.position.x() << ", " << raw.position.y() << ", " << raw.position.z() << ")";

                // 真の軌道のデータも表示
                if (!trajectory_data_->ground_truth.empty() &&
                    selected_frame_index_ < static_cast<int>(trajectory_data_->ground_truth.size()))
                {
                    const auto &gt = trajectory_data_->ground_truth[selected_frame_index_];
                    oss << " | GT Pos: (" << gt.position.x() << ", " << gt.position.y() << ", " << gt.position.z() << ")";
                }

                oss << " | Filtered Pos: (" << filtered.position.x() << ", " << filtered.position.y() << ", " << filtered.position.z() << ")";
            }

            info_text = oss.str();
        }
        else
        {
            info_text = "No trajectory data";
        }

        drawInfoText(10, 20, info_text);

        // 姿勢角（オイラー角）情報の表示
        if (!trajectory_data_->raw_trajectory.empty() &&
            !trajectory_data_->filtered_trajectory.empty() &&
            !trajectory_data_->ground_truth.empty() &&
            selected_frame_index_ < static_cast<int>(trajectory_data_->raw_trajectory.size()) &&
            selected_frame_index_ < static_cast<int>(trajectory_data_->filtered_trajectory.size()) &&
            selected_frame_index_ < static_cast<int>(trajectory_data_->ground_truth.size()))
        {

            const Eigen::Quaterniond &q_raw = trajectory_data_->raw_trajectory[selected_frame_index_].orientation;
            const Eigen::Quaterniond &q_filtered = trajectory_data_->filtered_trajectory[selected_frame_index_].orientation;
            const Eigen::Quaterniond &q_gt = trajectory_data_->ground_truth[selected_frame_index_].orientation;

            drawEulerAngles(q_raw, q_filtered, q_gt);
        }

        // コントロール情報を表示
        drawInfoText(10, glutGet(GLUT_WINDOW_HEIGHT) - 20,
                     "Controls: r-Raw Traj, f-Filtered Traj, c-Frames, g-Grid, a-Animate, +/- Zoom, s-Save");

        glutSwapBuffers();
    }

    // アニメーションフレームを進める
    void advanceFrame()
    {
        if (!trajectory_data_->raw_trajectory.empty())
        {
            selected_frame_index_ = (selected_frame_index_ + animation_speed_) % trajectory_data_->raw_trajectory.size();
            glutPostRedisplay();
        }
    }

    // キーボード入力を処理
    void handleKeyboard(unsigned char key, int, int)
    {
        switch (key)
        {
        case 'r': // 生データの表示/非表示を切り替え
            show_raw_ = !show_raw_;
            break;
        case 'f': // フィルタリング済みデータの表示/非表示を切り替え
            show_filtered_ = !show_filtered_;
            break;
        case 't': // 真の軌道の表示/非表示を切り替え
            show_ground_truth_ = !show_ground_truth_;
            break;
        case 'c': // 座標フレームの表示/非表示を切り替え
            show_coordinate_frames_ = !show_coordinate_frames_;
            break;
        case 'g': // グリッドの表示/非表示を切り替え
            show_ground_plane_ = !show_ground_plane_;
            break;
        case '+': // カメラを近づける
            camera_distance_ *= 0.9f;
            break;
        case '-': // カメラを遠ざける
            camera_distance_ *= 1.1f;
            break;
        case 'a': // アニメーションを開始/停止
            animation_running_ = !animation_running_;
            if (animation_running_)
            {
                glutTimerFunc(100, timerCallback, 0);
            }
            break;
        case '1': // アニメーション速度を下げる
            if (animation_speed_ > 1)
                animation_speed_--;
            break;
        case '2': // アニメーション速度を上げる
            animation_speed_++;
            break;
        case 's': // 軌道データを保存
            trajectory_data_->saveToFile("trajectory_data.csv");
            break;
        case 'q': // 終了
        case 27:  // ESC
            exit(0);
            break;
        }
        glutPostRedisplay();
    }

    // 特殊キー入力を処理
    void handleSpecialKeys(int key, int, int)
    {
        switch (key)
        {
        case GLUT_KEY_LEFT: // 前のフレーム
            if (selected_frame_index_ > 0)
                selected_frame_index_--;
            break;
        case GLUT_KEY_RIGHT: // 次のフレーム
            if (!trajectory_data_->raw_trajectory.empty() &&
                selected_frame_index_ < static_cast<int>(trajectory_data_->raw_trajectory.size()) - 1)
            {
                selected_frame_index_++;
            }
            break;
        case GLUT_KEY_UP: // カメラ角度を上に
            camera_phi_ += 0.05f;
            if (camera_phi_ > M_PI / 2 - 0.1f)
                camera_phi_ = M_PI / 2 - 0.1f;
            break;
        case GLUT_KEY_DOWN: // カメラ角度を下に
            camera_phi_ -= 0.05f;
            if (camera_phi_ < -M_PI / 2 + 0.1f)
                camera_phi_ = -M_PI / 2 + 0.1f;
            break;
        }
        glutPostRedisplay();
    }

    // マウス入力を処理
    void handleMouse(int button, int state, int x, int y)
    {
        if (button == GLUT_LEFT_BUTTON)
        {
            if (state == GLUT_DOWN)
            {
                // 左ボタンが押されたときの処理
            }
        }
    }

    // マウス移動を処理
    void handleMouseMotion(int x, int y)
    {
        static int last_x = x;
        static int last_y = y;

        const int dx = x - last_x;
        const int dy = y - last_y;

        // 左ボタンドラッグでカメラ回転
        if (glutGetModifiers() & GLUT_ACTIVE_SHIFT)
        {
            camera_target_.x() += dx * 0.01f;
            camera_target_.z() -= dy * 0.01f;
        }
        else
        {
            camera_theta_ -= dx * 0.01f;
            camera_phi_ += dy * 0.01f;

            // 制限を設ける
            if (camera_phi_ > M_PI / 2 - 0.1f)
                camera_phi_ = M_PI / 2 - 0.1f;
            if (camera_phi_ < -M_PI / 2 + 0.1f)
                camera_phi_ = -M_PI / 2 + 0.1f;
        }

        last_x = x;
        last_y = y;

        glutPostRedisplay();
    }

    // ウィンドウのリサイズを処理
    void handleReshape(int width, int height)
    {
        glViewport(0, 0, width, height);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(45.0f, static_cast<float>(width) / height, 0.1f, 100.0f);
        glMatrixMode(GL_MODELVIEW);
    }

    // アニメーションタイマーコールバック
    static void timerCallback(int)
    {
        TrajectoryVisualizer *instance = getInstance();
        if (instance && instance->animation_running_)
        {
            instance->advanceFrame();
            glutTimerFunc(100, timerCallback, 0);
        }
    }

    // シングルトンインスタンス変数
    static TrajectoryVisualizer *instance_;

    // シングルトンインスタンスを取得（glutコールバック用）
    static TrajectoryVisualizer *getInstance()
    {
        if (!instance_)
        {
            std::cerr << "Error: TrajectoryVisualizer instance not set\n";
        }
        return instance_;
    }

    // シングルトンインスタンスを設定
    static void setInstance(TrajectoryVisualizer *vis)
    {
        instance_ = vis;
    }
};

// 静的メンバ変数の定義
TrajectoryVisualizer *TrajectoryVisualizer::instance_ = nullptr;

// カルマンフィルタのテストを実行し、結果を可視化
void runFilteringTest()
{
    const int seed = 12345; // 乱数シード
    std::srand(seed);

    // EKFのインスタンスを作成
    const double dt = 0.1; // タイムステップ
    PoseEKF ekf(dt);

    // トラジェクトリデータ
    TrajectoryData trajectory_data;

    // 初期状態
    const Eigen::Vector3d initial_position(0.0, 0.0, 0.0);
    const Eigen::Vector3d initial_velocity(0.5, 0.0, 0.3);
    const Eigen::Vector3d initial_angular_velocity(0.0, 0.2, 0.3);
    const Eigen::Quaterniond initial_orientation(1.0, 0.0, 0.0, 0.0);

    ekf.initialize(initial_position, initial_orientation, initial_velocity, initial_angular_velocity);

    // 真の軌道（シミュレーション）
    std::vector<Eigen::Vector3d> true_positions;
    std::vector<Eigen::Quaterniond> true_orientations;

    // 8の字を描く軌道を生成
    const int num_steps = 100;
    for (int i = 0; i < num_steps; ++i)
    {
        const double t = static_cast<double>(i) / num_steps * 2.0 * M_PI;

        // 8の字の軌道座標
        const Eigen::Vector3d true_pos(
            4.0 * sin(t),
            0.5 * sin(2.0 * t),
            2.0 * sin(2.0 * t));

        // 進行方向を向く回転
        Eigen::Vector3d direction;
        if (i > 0)
        {
            direction = (true_pos - true_positions.back()).normalized();
        }
        else
        {
            direction = Eigen::Vector3d(1, 0, 0);
        }

        // 回転軸と角度を計算
        const Eigen::Vector3d up(0, 1, 0);
        const Eigen::Vector3d right = up.cross(direction).normalized();

        // 回転クォータニオンを作成
        Eigen::Quaterniond true_ori = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d(1, 0, 0), direction);

        // 軌道を保存
        true_positions.push_back(true_pos);
        true_orientations.push_back(true_ori);

        // 真の軌道データをTrajectoryDataに追加
        trajectory_data.ground_truth.emplace_back(i * 1.0, true_pos, true_ori, false);
    }

    // ノイズを加えた観測データを生成
    std::vector<Eigen::Vector3d> noisy_positions;
    std::vector<Eigen::Quaterniond> noisy_orientations;

    const double pos_noise_level = 0.2;  // 位置ノイズレベル
    const double ori_noise_level = 0.05; // 向きノイズレベル

    for (int i = 0; i < num_steps; ++i)
    {
        // 位置にノイズを追加
        const Eigen::Vector3d noise_pos = Eigen::Vector3d::Random() * pos_noise_level;
        const Eigen::Vector3d noisy_pos = true_positions[i] + noise_pos;

        // 向きにノイズを追加
        const Eigen::Vector3d noise_axis = Eigen::Vector3d::Random().normalized();
        const double noise_angle = static_cast<double>(rand()) / RAND_MAX * ori_noise_level;
        const Eigen::Quaterniond noise_quat(Eigen::AngleAxisd(noise_angle, noise_axis));
        const Eigen::Quaterniond noisy_ori = true_orientations[i] * noise_quat;

        // ノイズ付きデータを保存
        noisy_positions.push_back(noisy_pos);
        noisy_orientations.push_back(noisy_ori);

        // トラジェクトリデータにノイズ付き観測を追加
        trajectory_data.raw_trajectory.emplace_back(i * 1.0, noisy_pos, noisy_ori, false);
    }

    // プロセスノイズを調整 (小さくすると強くフィルタ)
    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(13, 13);
    Q.block<3, 3>(0, 0) *= 0.05;       // 位置のプロセスノイズ
    Q.block<3, 3>(3, 3) *= 0.2;        // 速度のプロセスノイズ
    Q.block<4, 4>(6, 6) *= 0.02;       // クォータニオンのプロセスノイズ
    Q.block<3, 3>(10, 10) *= 0.3;      // 角速度のプロセスノイズ
    ekf.setProcessNoise(Q);

    // 観測ノイズを調整 (大きくすると強くフィルタ)
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(7, 7);
    R.block<3, 3>(0, 0) *= 0.1;        // 位置観測のノイズ
    R.block<4, 4>(3, 3) *= 0.05;       // クォータニオン観測のノイズ
    ekf.setMeasurementNoise(R);

    // カルマンフィルタでフィルタリング
    Eigen::Vector3d filtered_position, filtered_velocity, filtered_angular_velocity;
    Eigen::Quaterniond filtered_orientation;

    for (int i = 0; i < num_steps; ++i)
    {
        // 予測ステップ
        // ekf.advancedPredict(10);
        ekf.predict();

        // 更新ステップ
        ekf.update(noisy_positions[i], noisy_orientations[i]);

        // フィルタリング結果を取得
        ekf.getState(filtered_position, filtered_orientation, filtered_velocity, filtered_angular_velocity);

        // フィルタリング結果をトラジェクトリデータに追加
        trajectory_data.filtered_trajectory.emplace_back(i * 1.0, filtered_position, filtered_orientation, true);
    }

    // 可視化の初期化と実行
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(1024, 768);
    glutCreateWindow("EKF Trajectory Visualization");

    // OpenGLの初期設定
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.1f, 0.1f, 0.2f, 1.0f);

    // 可視化クラスの作成
    TrajectoryVisualizer visualizer(&trajectory_data);
    TrajectoryVisualizer::setInstance(&visualizer);

    // GLUTコールバックの設定
    glutDisplayFunc([]()
                    { TrajectoryVisualizer::getInstance()->render(); });
    glutKeyboardFunc([](unsigned char key, int x, int y)
                     { TrajectoryVisualizer::getInstance()->handleKeyboard(key, x, y); });
    glutSpecialFunc([](int key, int x, int y)
                    { TrajectoryVisualizer::getInstance()->handleSpecialKeys(key, x, y); });
    glutReshapeFunc([](int w, int h)
                    { TrajectoryVisualizer::getInstance()->handleReshape(w, h); });
    glutMouseFunc([](int button, int state, int x, int y)
                  { TrajectoryVisualizer::getInstance()->handleMouse(button, state, x, y); });
    glutMotionFunc([](int x, int y)
                   { TrajectoryVisualizer::getInstance()->handleMouseMotion(x, y); });

    // 使用方法の表示
    std::cout << "EKF Trajectory Visualization Controls:\n";
    std::cout << "  r - Toggle raw trajectory\n";
    std::cout << "  f - Toggle filtered trajectory\n";
    std::cout << "  t - Toggle ground truth trajectory\n";
    std::cout << "  c - Toggle coordinate frames\n";
    std::cout << "  g - Toggle ground plane grid\n";
    std::cout << "  Left/Right Arrows - Previous/Next frame\n";
    std::cout << "  Up/Down Arrows - Adjust camera angle\n";
    std::cout << "  Mouse Drag - Rotate camera\n";
    std::cout << "  Shift+Mouse Drag - Pan camera\n";
    std::cout << "  +/- - Zoom in/out\n";
    std::cout << "  a - Start/stop animation\n";
    std::cout << "  1/2 - Decrease/Increase animation speed\n";
    std::cout << "  s - Save trajectory data to file\n";
    std::cout << "  q/ESC - Quit\n";

    // メインループ開始
    glutMainLoop();
}

int main(int argc, char **argv)
{
    glutInit(&argc, argv);
    srand(time(nullptr));

    runFilteringTest();

    return 0;
}
