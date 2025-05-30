import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2
import os
from tqdm import tqdm

def create_video_with_effort(file_path, output_video="effort_visualization.mp4", fps=10):
    """
    读取HDF5文件中的相机图像和effort数据，并创建可视化视频
    
    参数:
        file_path: HDF5文件的路径
        output_video: 输出视频的路径
        fps: 视频的帧率
    """
    try:
        # 打开HDF5文件
        with h5py.File(file_path, 'r') as f:
            # 检查并读取左手腕相机图像
            if 'observations' in f and 'images' in f['observations'] and 'cam_left_wrist' in f['observations']['images']:
                wrist_images = f['observations']['images']['cam_left_wrist'][:]
            else:
                print("无法找到左手腕相机图像数据。文件结构如下:")
                f.visititems(lambda name, obj: print(f" - {'数据集' if isinstance(obj, h5py.Dataset) else '组'}: {name}"))
                return
                
            # 检查并读取effort数据
            if 'observations' in f and 'effort' in f['observations']:
                effort_data = f['observations']['effort'][:]
            else:
                print("无法找到effort数据。")
                return
            
            # 确保图像和effort数据长度匹配
            n_frames = min(len(wrist_images), len(effort_data))
            print(f"找到 {n_frames} 帧数据")
            
            if n_frames == 0:
                print("没有可用的帧")
                return
                
            # 获取图像尺寸和effort维度
            frame_height, frame_width = wrist_images[0].shape[:2]
            n_effort_dims = effort_data.shape[1]
            
            if n_effort_dims != 14:
                print(f"警告: effort数据不是14维，而是{n_effort_dims}维")
            
            # 创建临时目录存储帧
            temp_dir = "temp_frames"
            os.makedirs(temp_dir, exist_ok=True)
            
            # 设置视频编码器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或尝试 'avc1'
            video_height = frame_height + 400  # 为effort图表留出空间
            video_out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, video_height))
            
            print("正在生成视频帧...")
            
            # 为每一帧创建可视化图像
            for i in tqdm(range(n_frames)):
                # 创建图像
                fig = plt.figure(figsize=(frame_width/100, video_height/100), dpi=100)
                
                # 上半部分显示相机图像
                ax_img = plt.subplot2grid((5, 1), (0, 0), rowspan=2)
                ax_img.imshow(wrist_images[i])
                ax_img.set_title(f'左手腕相机图像 - 帧 {i+1}/{n_frames}')
                ax_img.axis('off')
                
                # 下半部分显示effort数据
                gs = GridSpec(3, 5, bottom=0, top=0.5)
                
                for j in range(min(n_effort_dims, 14)):
                    ax = plt.subplot(gs[j // 5, j % 5])
                    
                    # 绘制effort数据的时间序列（显示前后10帧）
                    start_idx = max(0, i - 10)
                    end_idx = min(n_frames, i + 11)
                    time_range = np.arange(start_idx, end_idx)
                    
                    ax.plot(time_range, effort_data[start_idx:end_idx, j], 'b-')
                    # 标记当前点
                    ax.plot(i, effort_data[i, j], 'ro')
                    
                    ax.set_title(f'维度 #{j+1}', fontsize=8)
                    ax.set_ylim([-3, 3])
                    ax.grid(True)
                    
                    # 仅在最下面一行显示x轴标签
                    if j // 5 == 2 or (j // 5 == 1 and j % 5 >= 3):
                        ax.set_xlabel('帧', fontsize=8)
                    else:
                        ax.set_xticklabels([])
                    
                    # 仅在最左侧显示y轴标签
                    if j % 5 == 0:
                        ax.set_ylabel('数值', fontsize=8)
                    else:
                        ax.set_yticklabels([])
                    
                    # 精简刻度标签
                    ax.tick_params(axis='both', which='major', labelsize=6)
                
                plt.tight_layout()
                
                # 保存当前帧为图像
                frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
                plt.savefig(frame_path)
                plt.close()
                
                # 读取保存的图像并添加到视频
                frame = cv2.imread(frame_path)
                video_out.write(frame)
            
            # 释放视频编写器
            video_out.release()
            
            # 删除临时文件
            for i in range(n_frames):
                frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
                if os.path.exists(frame_path):
                    os.remove(frame_path)
            
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
            
            print(f"视频已成功生成: {output_video}")
            
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 修改这里的文件路径为你的HDF5文件路径
    hdf5_file_path = "/baai-cwm-nas/public_data/scenes/ych_newpick/demo/episode_43.hdf5"  # 替换为实际的文件路径
    
    create_video_with_effort(hdf5_file_path)