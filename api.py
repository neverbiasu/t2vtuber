from gradio_client import Client


class ImageProcessingClient:
    def __init__(self, base_url):
        self.client = Client(base_url)

    def load_model(self, model_id):
        result = self.client.predict(model_id, fn_index=0)  # 模型 ID
        print("Model loaded:", result)
        return result

    def segment_image(self, input_image_url, model_id, anime_style):
        result = self.client.predict(
            input_image_url,  # 输入图像 URL
            model_id,  # 模型 ID
            input_image_url,  # Segment Anything 图像 URL
            anime_style,  # 动漫风格
            fn_index=4,
        )
        print("Image segmented:", result)
        return result

    def generate_selected_mask(self, input_image_url, selected_mask_url):
        result = self.client.predict(
            input_image_url,  # 输入图像 URL
            selected_mask_url,  # 选中的掩码图像 URL
            fn_index=10,
        )
        print("Selected mask generated:", result)
        return result

    def inpaint_image(
        self,
        input_image_url,
        selected_mask_url,
        prompt,
        negative_prompt,
        sampling_steps,
        guidance_scale,
        seed,
        model_id,
        save_mask,
        mask_area_only,
        sampler,
        iterations,
    ):
        result = self.client.predict(
            input_image_url,  # 输入图像 URL
            selected_mask_url,  # 选中的掩码图像 URL
            prompt,  # 修复提示
            negative_prompt,  # 负面提示
            sampling_steps,  # 采样步骤
            guidance_scale,  # 指导尺度
            seed,  # 随机种子
            model_id,  # 修复模型 ID
            save_mask,  # 保存掩码
            mask_area_only,  # 仅掩码区域
            sampler,  # 采样器
            iterations,  # 迭代次数
            fn_index=14,
        )
        print("Image inpainted:", result)
        return result


# 使用示例
if __name__ == "__main__":
    base_url = "http://127.0.0.1:7860/"
    client = ImageProcessingClient(base_url)

    # 第一步：加载模型
    model_id = "sam2_hiera_large.pt,sam2_hiera_large.pt"
    client.load_model(model_id)

    # 第二步：调用 SAM 的分割预测
    input_image_url = "https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png"
    anime_style = True
    client.segment_image(input_image_url, model_id, anime_style)

    # 第三步：生成选中的掩码
    selected_mask_url = "https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png"
    client.generate_selected_mask(input_image_url, selected_mask_url)

    # 第四步：生成风格化的 texture 图片
    prompt = "Howdy!"
    negative_prompt = "Howdy!"
    sampling_steps = 1
    guidance_scale = 0.1
    seed = -1
    inpainting_model_id = "stabilityai/stable-diffusion-2-inpainting,stabilityai/stable-diffusion-2-inpainting"
    save_mask = True
    mask_area_only = True
    sampler = "DDIM,DDIM"
    iterations = 1
    client.inpaint_image(
        input_image_url,
        selected_mask_url,
        prompt,
        negative_prompt,
        sampling_steps,
        guidance_scale,
        seed,
        inpainting_model_id,
        save_mask,
        mask_area_only,
        sampler,
        iterations,
    )
