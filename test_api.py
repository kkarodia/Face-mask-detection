"""
API Test Client

Simple script to test the Face Mask Detection API endpoints.
"""

import requests
import argparse
from pathlib import Path
import json
import time


def test_health_check(base_url: str):
    """Test the health check endpoint."""
    print("\n" + "="*60)
    print("Testing Health Check Endpoint")
    print("="*60)
    
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_detect_endpoint(base_url: str, image_path: str):
    """Test the mask detection endpoint."""
    print("\n" + "="*60)
    print("Testing Mask Detection Endpoint")
    print("="*60)
    print(f"Image: {image_path}")
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            
            start_time = time.time()
            response = requests.post(f"{base_url}/detect", files=files)
            total_time = (time.time() - start_time) * 1000
            
        print(f"Status Code: {response.status_code}")
        print(f"Total Request Time: {total_time:.2f} ms")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\nPrediction Result:")
            print(f"  - Prediction: {result['prediction']}")
            print(f"  - Confidence: {result['confidence']:.4f}")
            print(f"  - Message: {result['message']}")
            print(f"  - Inference Time: {result['inference_time_ms']:.2f} ms")
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_detect_batch_endpoint(base_url: str, image_dir: str):
    """Test the batch mask detection endpoint."""
    print("\n" + "="*60)
    print("Testing Batch Mask Detection Endpoint")
    print("="*60)
    
    try:
        # Get image files
        image_dir = Path(image_dir)
        image_files = list(image_dir.glob("*.jpg"))[:5]  # Limit to 5 images
        
        print(f"Processing {len(image_files)} images")
        
        # Prepare files
        files = [('files', (img.name, open(img, 'rb'), 'image/jpeg')) 
                for img in image_files]
        
        start_time = time.time()
        response = requests.post(f"{base_url}/detect-batch", files=files)
        total_time = (time.time() - start_time) * 1000
        
        # Close files
        for _, file_tuple in files:
            file_tuple[1].close()
        
        print(f"Status Code: {response.status_code}")
        print(f"Total Request Time: {total_time:.2f} ms")
        
        if response.status_code == 200:
            results = response.json()['results']
            print(f"\nBatch Results:")
            for i, result in enumerate(results):
                if 'error' in result:
                    print(f"  {i+1}. {result['filename']}: ERROR - {result['error']}")
                else:
                    print(f"  {i+1}. {result['filename']}:")
                    print(f"      Prediction: {result['prediction']}")
                    print(f"      Confidence: {result['confidence']:.4f}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_model_info(base_url: str):
    """Test the model info endpoint."""
    print("\n" + "="*60)
    print("Testing Model Info Endpoint")
    print("="*60)
    
    try:
        response = requests.get(f"{base_url}/model-info")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            info = response.json()
            print(f"\nModel Information:")
            print(f"  - Architecture: {info['model_architecture']}")
            print(f"  - Input Shape: {info['input_shape']}")
            print(f"  - Output Classes: {info['output_classes']}")
            print(f"  - Total Parameters: {info['total_parameters']:,}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description='Test Face Mask Detection API')
    parser.add_argument('--url', type=str, default='http://localhost:8000',
                       help='API base URL')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to test image')
    parser.add_argument('--image-dir', type=str, default=None,
                       help='Path to directory with test images')
    parser.add_argument('--all', action='store_true',
                       help='Run all tests')
    
    args = parser.parse_args()
    
    base_url = args.url.rstrip('/')
    
    print("="*60)
    print("FACE MASK DETECTION API TEST SUITE")
    print("="*60)
    print(f"API URL: {base_url}")
    
    # Test health check
    health_ok = test_health_check(base_url)
    
    if not health_ok:
        print("\n Health check failed. API may not be running.")
        return
    
    print("\n Health check passed!")
    
    # Test model info
    if args.all or not args.image:
        info_ok = test_model_info(base_url)
        if info_ok:
            print("\n Model info test passed!")
    
    # Test detect endpoint
    if args.image:
        detect_ok = test_detect_endpoint(base_url, args.image)
        if detect_ok:
            print("\n Detection test passed!")
        else:
            print("\n Detection test failed!")
    
    # Test batch endpoint
    if args.image_dir:
        batch_ok = test_detect_batch_endpoint(base_url, args.image_dir)
        if batch_ok:
            print("\n Batch detection test passed!")
        else:
            print("\n Batch detection test failed!")
    
    print("\n" + "="*60)
    print("TEST SUITE COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()
