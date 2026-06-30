#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
import os
_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'sft'))

import requests
from typing import List, Dict, Any, Optional


def send_batch_request(
    data: List[List[List[float]]],
    checkpoint: str,
    server_url: str = "http://127.0.0.1:10089/cls",
    timeout: int = 600,
    **model_config
) -> Optional[Dict[str, Any]]:

    request_body = {
        "data": data,
        "checkpoint": checkpoint,
        **model_config
    }
    
    try:
        response = requests.post(
            server_url,
            json=request_body,
            timeout=timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("status") == "success":
                return result
            else:
                print(f"❌ Server returned error: {result.get('error')}")
                return None
        else:
            print(f"❌ HTTP error: {response.status_code}")
            print(f"   {response.text[:200]}")
            return None
            
    except requests.exceptions.Timeout:
        print(f"⏰ Request timeout after {timeout}s")
        return None
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to server: {server_url}")
        return None
    except Exception as e:
        print(f"❌ Request error: {e}")
        return None
