import os

package_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(package_dir, "../")

if not os.path.exists(data_dir):
    raise Exception(f"Data directory not found: {data_dir}")
