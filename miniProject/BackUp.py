import os
import shutil
import datetime
import schedule
import time

source_dir ="C:/VideoDownload/Test/Source"
destination_dir="C:/VideoDownload/Test/BackUp"

def copy_folder_to_directory(source,dest):
    today=datetime.date.today()
    dest_dir=os.path.join(dest,str(today))
    
    try:
        shutil.copytree(source,dest_dir)
        print(f"Folder copied to {dest_dir}")
    except FileExistsError:
        print(f"Folder Already  Exist in :{dest}")

schedule.every().day.at("23:09").do(lambda: copy_folder_to_directory(source_dir,destination_dir))


while True:
    schedule.run_pending()
    time.sleep(60)
    