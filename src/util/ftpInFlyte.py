# using FPT to
# download data and model files from workspace on server to local flyte
# #upload the output from local flyte env to worksapce server

import datetime
import logging
import mimetypes
import os
import sys
import zipfile
from ftplib import error_perm

from integrations.lsf import open_ftp_connection

logger = logging.getLogger(__name__)

LOCAL_MODEL_PATH = "/workspace/tmp"
LOCAL_DATA_PATH = "/workspace/tmp"


def get_file_type(file_path):
    # Get the file extension
    _, ext = os.path.splitext(file_path)

    # Get the MIME type based on the file extension
    mime_type, _ = mimetypes.guess_type(file_path)

    return ext, mime_type


def insert_timestamp(filename):
    # Get the current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    # Split the filename into name and extension
    name, ext = filename.rsplit(".", 1)

    # Insert the timestamp into the filename
    new_filename = f"{name}_{timestamp}.{ext}"

    return new_filename


def count_files_in_directory(directory):
    return len(
        [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    )


def download_folder(remote_file_folder, local_file_folder):
    os.makedirs(local_file_folder, exist_ok=True)
    logger.info(f"local folder start '{local_file_folder}'.")

    local_folder = local_file_folder

    with open_ftp_connection() as ftp:
        model_name = os.path.basename(remote_file_folder)
        if remote_file_folder.endswith("/"):
            model_name = os.path.basename(remote_file_folder[:-1])

        local_folder = os.path.join(local_file_folder, model_name)
        os.makedirs(local_folder, exist_ok=True)

        ftp.cwd(remote_file_folder)

        for name in ftp.nlst():
            remote_path = os.path.join(remote_file_folder, name)
            local_path = os.path.join(local_folder, name)

            if name.endswith("/"):
                # Skip directories
                continue

            # Download file
            with open(local_path, "wb") as f:
                ftp.retrbinary("RETR " + remote_path, f.write, 1024)
                logger.info(
                    f"file downloaded from '{remote_path}' are stored in '{local_path}'."
                )

        file_num = count_files_in_directory(local_folder)

        logger.info(
            f"{file_num}files downloaded from '{remote_file_folder}' are stored in '{local_file_folder}'."
        )

    return local_folder


def download_file(remote_file, local_folder):
    # Check if local file path is valid
    filename = os.path.basename(remote_file)

    os.makedirs(local_folder, exist_ok=True)
    local_path = os.path.join(local_folder, filename)
    if not os.path.exists(local_path):
        # File does not exist, create it
        open(local_path, "x").close()

    with open_ftp_connection() as ftp:
        with open(local_path, "wb") as f:
            ftp.retrbinary("RETR " + remote_file, f.write, 1024)

    logger.info(f"Downloaded {remote_file} to {local_path}")

    return local_path


def upload_file(remote_folder, local_file):
    # get the upload filename
    filename = os.path.basename(local_file)

    # insert time stamp into remote upload filename
    remote_name = insert_timestamp(filename)
    remote_path = remote_folder + "/" + remote_name

    with open_ftp_connection() as ftp:
        with open(local_file, "rb") as f:
            ftp.storbinary(f"STOR {remote_path}", f)  # upload to ftp

    return remote_path


def upload_all_files(remote_folder, local_folder):
    # Initialize a list to store the remote paths
    remote_paths = []

    with open_ftp_connection() as ftp:
        # Walk through the local folder and its sub-folders
        for dirpath, dirnames, filenames in os.walk(local_folder):
            for filename in filenames:
                # Get the local file path
                local_path = os.path.join(dirpath, filename)

                # Get the remote file path
                remote_path = os.path.join(
                    remote_folder, os.path.relpath(local_path, local_folder)
                )

                # Create remote directory if it doesn't exist
                remote_dir = os.path.dirname(remote_path)
                print(f"{remote_dir}")
                try:
                    ftp.mkd(remote_dir)
                except error_perm as e:
                    # Ignore the error if the directory already exists
                    if not str(e).startswith("550"):
                        raise

                # Upload the file
                print(f"local path {local_path}->remote {remote_path}")
                with open(local_path, "rb") as f:
                    ftp.storbinary(f"STOR {remote_path}", f)  # upload to ftp

                remote_paths.append(remote_path)
            # end for filename in filenames
    return remote_paths


def list_files_and_dirs(directory):
    return [
        os.path.join(dp, f)
        for dp, dn, filenames in os.walk(directory)
        for f in filenames
    ]


def get_cached_model(remote_model_folder: str) -> str:
    l_folder = "/workspace/tmp/model"
    local_folder = download_folder(remote_model_folder, l_folder)

    logger.info(f"local model folder is {local_folder}")
    # return FlyteDirectory(path=str(local_folder))  # relative_model_dir
    return local_folder


def get_raw_data(remote_file: str) -> str:
    l_folder = "/workspace/tmp"
    # l_folder = os.getcwd() + "/tmp"
    local_file = download_file(remote_file, l_folder)
    logger.info(f"local data file is {local_file}")

    # return FlyteFile(path=local_file)
    return local_file


def zip_folder(file_name: str, target_dir: str) -> str:
    zip_file_name = file_name + ".zip"
    zipobj = zipfile.ZipFile(zip_file_name, "w", zipfile.ZIP_DEFLATED)
    rootlen = len(target_dir) + 1
    for base, dirs, files in os.walk(target_dir):
        for file in files:
            fn = os.path.join(base, file)
            zipobj.write(fn, fn[rootlen:])
    return zip_file_name


if __name__ == "__main__":
    # e.g. python ./util/ftpInFlyte.py insert_timestamp
    if len(sys.argv) > 1:
        if sys.argv[1] == "get_file_type":
            ext, mimetype = get_file_type(
                "/work/nsd_aiml/users/beizhang/workspace/Finetuning/data/sample_cfam_cb007487_cp1.txt"
            )
            print(f"ext:{ext} mimetype:{mimetype}")
        elif sys.argv[1] == "insert_timestamp":
            new_file = insert_timestamp("sample_cfam_cb007487_cp1.txt")
            print(new_file)
        elif sys.argv[1] == "relative_path":
            cwd = os.getcwd()
            model_dir = "/workspace/firstproject/firstproject/tmp/flan-t5-base"
            # Calculate relative path
            relative_model_dir = "./" + os.path.relpath(model_dir, cwd)
            print(relative_model_dir)
        elif sys.argv[1] == "upload_file":
            remote_path = "/work/nsd_aiml/users/beizhang/workspace/temp"
            local_file = "./data/ft_data.json"
            if not os.path.exists(local_file):
                print(f"{local_file} is not existed\n")
            else:
                upload_file(remote_path, local_file)
        else:
            remote_path = "/work/nsd_aiml/users/beizhang/workspace/temp"
            local_path = "./data"
            upload_all_files(remote_path, local_path)
    else:
        print("give one of following command:\n ")
        print()
