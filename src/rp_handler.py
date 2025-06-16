import runpod
from runpod.serverless.utils import rp_upload
import json
import urllib.request
import urllib.parse
import time
import os
import requests
import base64
from io import BytesIO
import oss2

# Time to wait between API check attempts in milliseconds
COMFY_API_AVAILABLE_INTERVAL_MS = 50
# Maximum number of API check attempts
COMFY_API_AVAILABLE_MAX_RETRIES = 500
# Time to wait between poll attempts in milliseconds
COMFY_POLLING_INTERVAL_MS = int(os.environ.get("COMFY_POLLING_INTERVAL_MS", 250))
# Maximum number of poll attempts
COMFY_POLLING_MAX_RETRIES = int(os.environ.get("COMFY_POLLING_MAX_RETRIES", 500))
# Host where ComfyUI is running
COMFY_HOST = "127.0.0.1:8188"
# Enforce a clean state after each job is done
# see https://docs.runpod.io/docs/handler-additional-controls#refresh-worker
REFRESH_WORKER = os.environ.get("REFRESH_WORKER", "false").lower() == "true"

# 获取阿里云相关环境变量
ALIYUN_ACCESS_KEY_ID = os.environ.get('ALIYUN_ACCESS_KEY_ID')
ALIYUN_ACCESS_KEY_SECRET = os.environ.get('ALIYUN_ACCESS_KEY_SECRET')
ALIYUN_ENDPOINT = os.environ.get('ALIYUN_ENDPOINT')
ALIYUN_BUCKET_NAME = os.environ.get('ALIYUN_BUCKET_NAME')
# 检查环境变量是否存在
if not all([ALIYUN_ACCESS_KEY_ID, ALIYUN_ACCESS_KEY_SECRET, ALIYUN_ENDPOINT, ALIYUN_BUCKET_NAME]):
    raise ValueError("Missing required environment variables for Aliyun OSS")

# 阿里云账号AccessKey拥有所有API的访问权限，风险很高。强烈建议您创建并使用RAM用户进行API访问或日常运维，请登录RAM控制台创建RAM用户。
auth = oss2.Auth(ALIYUN_ACCESS_KEY_ID, ALIYUN_ACCESS_KEY_SECRET)
# Endpoint以杭州为例，其它Region请按实际情况填写。
bucket = oss2.Bucket(auth, ALIYUN_ENDPOINT, ALIYUN_BUCKET_NAME)

def upload_to_aliyun(local_file_path, oss_file_path):
    """
    上传文件到阿里云OSS
    :param local_file_path: 本地文件路径
    :param job_id: job的ID，用于生成OSS目标文件路径
    :return: 上传成功返回文件的完整OSS路径，失败返回None
    """
    try:
        # 获取文件后缀
        # 生成OSS目标文件路径
        start_time = time.time()  # 记录开始时间        
        # 上传文件
        result = bucket.put_object_from_file(oss_file_path, local_file_path)
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算耗时        
        if result.status == 200:
            print(f"成功上传到阿里云，耗时{elapsed_time}")       
            # 返回https的完整OSS路径
            return f"https://{ALIYUN_BUCKET_NAME}.{ALIYUN_ENDPOINT}/{oss_file_path}"
        else:
            return None
    except Exception as e:
        print(f"上传文件到阿里云OSS失败: {e}")
        return None
def validate_input(job_input):
    """
    Validates the input for the handler function.

    Args:
        job_input (dict): The input data to validate.

    Returns:
        tuple: A tuple containing the validated data and an error message, if any.
               The structure is (validated_data, error_message).
    """
    # Validate if job_input is provided
    if job_input is None:
        return None, "Please provide input"

    # Check if input is a string and try to parse it as JSON
    if isinstance(job_input, str):
        try:
            job_input = json.loads(job_input)
        except json.JSONDecodeError:
            return None, "Invalid JSON format in input"

    # Validate 'workflow' in input
    workflow = job_input.get("workflow")
    if workflow is None:
        return None, "Missing 'workflow' parameter"

    # Validate 'images' in input, if provided
    urls = job_input.get("urls")
    uploadFiles = job_input.get("uploadFiles")
    # Return validated data and no error
    return {"workflow": workflow, "urls": urls,"uploadFiles":uploadFiles}, None


def check_server(url, retries=500, delay=50):
    """
    Check if a server is reachable via HTTP GET request

    Args:
    - url (str): The URL to check
    - retries (int, optional): The number of times to attempt connecting to the server. Default is 50
    - delay (int, optional): The time in milliseconds to wait between retries. Default is 500

    Returns:
    bool: True if the server is reachable within the given number of retries, otherwise False
    """

    for i in range(retries):
        try:
            response = requests.get(url)

            # If the response status code is 200, the server is up and running
            if response.status_code == 200:
                print(f"runpod-worker-comfy - API is reachable")
                return True
        except requests.RequestException as e:
            # If an exception occurs, the server may not be ready
            pass

        # Wait for the specified delay before retrying
        time.sleep(delay / 1000)

    print(
        f"runpod-worker-comfy - Failed to connect to server at {url} after {retries} attempts."
    )
    return False

def download_file(urls):  
    for url_info in urls:
        try:
            name = url_info["name"]
            file_url = url_info["url"]
            download_path=url_info["path"]
            if not os.path.exists(download_path):
                os.makedirs(download_path)             
            start_time = time.time()  # 记录开始时间
            response = requests.get(file_url)
            end_time = time.time()  # 记录结束时间
            elapsed_time = end_time - start_time  # 计算耗时

            if response.status_code == 200:
                file_path = os.path.join(download_path, name)
                with open(file_path, 'wb') as file:
                    file.write(response.content)
                print(f"成功下载文件，耗时{elapsed_time}") 
            else:
                print(f"Failed to download {name}: HTTP Status Code {response.status_code}")
        except KeyError as e:
            print(f"Failed to download due to missing key: {e}")
        except Exception as e:
            print(f"Failed to download {name}: {str(e)}")

    return
def upload_images(images):
    """
    Upload a list of base64 encoded images to the ComfyUI server using the /upload/image endpoint.

    Args:
        images (list): A list of dictionaries, each containing the 'name' of the image and the 'image' as a base64 encoded string.
        server_address (str): The address of the ComfyUI server.

    Returns:
        list: A list of responses from the server for each image upload.
    """
    if not images:
        return {"status": "success", "message": "No images to upload", "details": []}

    responses = []
    upload_errors = []

    print(f"runpod-worker-comfy - image(s) upload")

    for image in images:
        name = image["name"]
        image_data = image["image"]
        blob = base64.b64decode(image_data)

        # Prepare the form data
        files = {
            "image": (name, BytesIO(blob), "image/png"),
            "overwrite": (None, "true"),
        }

        # POST request to upload the image
        response = requests.post(f"http://{COMFY_HOST}/upload/image", files=files)
        if response.status_code != 200:
            upload_errors.append(f"Error uploading {name}: {response.text}")
        else:
            responses.append(f"Successfully uploaded {name}")

    if upload_errors:
        print(f"runpod-worker-comfy - image(s) upload with errors")
        return {
            "status": "error",
            "message": "Some images failed to upload",
            "details": upload_errors,
        }

    print(f"runpod-worker-comfy - image(s) upload complete")
    return {
        "status": "success",
        "message": "All images uploaded successfully",
        "details": responses,
    }

def queue_workflow(workflow):
    """
    Queue a workflow to be processed by ComfyUI

    Args:
        workflow (dict): A dictionary containing the workflow to be processed

    Returns:
        dict: The JSON response from ComfyUI after processing the workflow
    """

    # The top level element "prompt" is required by ComfyUI
    data = json.dumps({"prompt": workflow}).encode("utf-8")

    req = urllib.request.Request(f"http://{COMFY_HOST}/prompt", data=data)
    return json.loads(urllib.request.urlopen(req).read())


def get_history(prompt_id):
    """
    Retrieve the history of a given prompt using its ID

    Args:
        prompt_id (str): The ID of the prompt whose history is to be retrieved

    Returns:
        dict: The history of the prompt, containing all the processing steps and results
    """
    with urllib.request.urlopen(f"http://{COMFY_HOST}/history/{prompt_id}") as response:
        return json.loads(response.read())


def base64_encode(img_path):
    """
    Returns base64 encoded image.

    Args:
        img_path (str): The path to the image

    Returns:
        str: The base64 encoded image
    """
    with open(img_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return f"{encoded_string}"


def process_output_result(outputs, job_id):
    # 如果 outputs 为空，直接返回错误信息
    if not outputs:
        return {
            "status": "error",
            "message": "outputs 为空，没有文件需要处理。"
        }

    # 获取 outputs.items() 的第一个元素
    node_id, node_output = next(iter(outputs.items()))

    # 检查是否有文件信息
    file_info_list = None
    if node_output:
        # 假设文件信息总是在 node_output 的第一个键对应的值中
        first_key = next(iter(node_output))
        file_info_list = node_output[first_key]

    if not file_info_list:
        return {
            "status": "error",
            "message": "未找到可处理的文件信息。"
        }

    # 获取第一个文件信息
    file_info = file_info_list[0]
    filename = file_info["filename"]
    subfolder = file_info["subfolder"]
    # The path where ComfyUI stores the generated images
    COMFY_OUTPUT_PATH = os.environ.get("COMFY_OUTPUT_PATH", "/comfyui/output")
    # expected image output folder
    local_file_path = os.path.join(COMFY_OUTPUT_PATH, subfolder, filename)

    # 检查文件是否存在
    if not os.path.exists(local_file_path):
        return {
            "status": "error",
            "message": f"文件 {local_file_path} 不存在。"
        }

    # The image is in the output folder
    if os.path.exists(local_file_path):
        # 使用 job_id 作为 OSS 文件路径的一部分，确保唯一性
        oss_file_path = f"{job_id}/{filename}"
        file_context = upload_to_aliyun(local_file_path, oss_file_path)
        print(
                "runpod-worker-comfy - the image was generated and uploaded to AWS S3"
        )       
        return {
            "status": "success",
            "message": file_context,
        }
    else:
        print("runpod-worker-comfy - the image does not exist in the output folder")
        return {
            "status": "error",
            "message": f"the image does not exist in the specified output folder: {local_file_path}",
        }


def handler(job):
    """
    The main function that handles a job of generating an image.

    This function validates the input, sends a prompt to ComfyUI for processing,
    polls ComfyUI for result, and retrieves generated images.

    Args:
        job (dict): A dictionary containing job details and input parameters.

    Returns:
        dict: A dictionary containing either an error message or a success status with generated images.
    """
    job_input = job["input"]

    # Make sure that the input is valid
    validated_data, error_message = validate_input(job_input)
    if error_message:
        return {"error": error_message}

    # Extract validated data
    workflow = validated_data["workflow"]
    #images = validated_data.get("images")
    urls=validated_data.get("urls")
    uploadFiles=validated_data.get("uploadFiles")

    # Make sure that the ComfyUI API is available
    check_server(
        f"http://{COMFY_HOST}",
        COMFY_API_AVAILABLE_MAX_RETRIES,
        COMFY_API_AVAILABLE_INTERVAL_MS,
    )

    # Upload images if they exist
    #upload_result = upload_images(images)
    download_file(urls);
    # Queue the workflow
    try:
        queued_workflow = queue_workflow(workflow)
        prompt_id = queued_workflow["prompt_id"]
        print(f"runpod-worker-comfy - queued workflow with ID {prompt_id}")
    except Exception as e:
        return {"error": f"Error queuing workflow: {str(e)}"}

    # Poll for completion
    print(f"runpod-worker-comfy - wait until image generation is complete")
    retries = 0
    try:
        while retries < COMFY_POLLING_MAX_RETRIES:
            history = get_history(prompt_id)

            # Exit the loop if we have found the history
            if prompt_id in history and history[prompt_id].get("outputs"):
                break
            else:
                # Wait before trying again
                time.sleep(COMFY_POLLING_INTERVAL_MS / 1000)
                retries += 1
        else:
            return {"error": "Max retries reached while waiting for image generation"}
    except Exception as e:
        return {"error": f"Error waiting for image generation: {str(e)}"}

    #生成的文件名称不固定，所有只能查询取到 
    files_result = process_output_result(history[prompt_id].get("outputs"), job["id"])
    upload_files_result = []
    #生成的文件名和输入的名字一样，可以直接操作,后面建议直接runpod的网盘操作，减少时间
    if uploadFiles:
        for file_info in uploadFiles:
            filename = file_info["filename"]
            path = file_info["path"]
            class_type=file_info["class_type"]
            # 构建本地文件路径
            local_file_path = os.path.join(path, filename)
            # 构建阿里云文件路径
            oss_file_path = filename
            # 调用上传函数
            file_url = upload_to_aliyun(local_file_path, oss_file_path)
            # 构建结果信息
            result_info = {
                "filename": filename,
                "file_url": file_url,
                "class_type":class_type
            }
            # 将结果信息添加到结果列表中
            upload_files_result.append(result_info)
    result = {**files_result, "refresh_worker": REFRESH_WORKER,"upload_files_result":upload_files_result}

    return result


# Start the handler only if this script is run directly
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
