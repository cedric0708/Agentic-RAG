import os
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from api.pharmacies import router

app = FastAPI()
app.include_router(router)
client = TestClient(app)
DOC = {"id": "123", "place_name": "테스트약국", "address_name": "지번 주소",
       "road_address_name": "도로명 주소", "x": "127.1", "y": "37.5", "distance": "123"}


class PharmacyTests(unittest.TestCase):
    def test_validation(self):
        for params in ({"lat": 91, "lng": 127}, {"lat": 37, "lng": 181},
                       {"lat": "nan", "lng": 127}, {"lat": 37, "lng": 127, "radius": 999},
                       {"lat": 37, "lng": 127, "page": 46}):
            self.assertEqual(client.get('/api/pharmacies', params=params).status_code, 422)
        self.assertEqual(client.get('/api/locations', params={"query": "  "}).status_code, 422)

    @patch('api.pharmacies.kakao_search', new_callable=AsyncMock)
    def test_mapping_and_pagination(self, search):
        search.return_value = {"documents": [DOC], "meta": {"is_end": False}}
        response = client.get('/api/pharmacies?lat=37.5&lng=127.1&page=2')
        self.assertEqual(response.status_code, 200)
        item = response.json()['items'][0]
        self.assertEqual((item['latitude'], item['longitude'], item['distance_m']), (37.5, 127.1, 123))
        self.assertEqual(item['address'], '도로명 주소')
        self.assertEqual(item['phone'], '')
        self.assertTrue(response.json()['has_next'])
        params = search.call_args.args[1]
        self.assertEqual((params['x'], params['y'], params['category_group_code'], params['sort']),
                         (127.1, 37.5, 'PM9', 'distance'))

    @patch('api.pharmacies.kakao_search', new_callable=AsyncMock)
    def test_locations_include_addresses(self, search):
        search.side_effect = [{"documents": [DOC]}, {"documents": [{"x": "127", "y": "37", "address_name": "서울"}]}]
        response = client.get('/api/locations?query=서울')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()['items']), 2)

    @patch('api.pharmacies.kakao_search', new_callable=AsyncMock)
    def test_empty_and_bad_upstream_data(self, search):
        search.return_value = {"documents": [], "meta": {"is_end": True}}
        self.assertEqual(client.get('/api/pharmacies?lat=37&lng=127').json()['has_next'], False)
        search.return_value = {"documents": [{}], "meta": {"is_end": True}}
        self.assertEqual(client.get('/api/pharmacies?lat=37&lng=127').status_code, 502)

    @patch.dict(os.environ, {"KAKAO_REST_API_KEY": ""})
    def test_missing_key(self):
        self.assertEqual(client.get('/api/pharmacies?lat=37&lng=127').status_code, 503)

    @patch.dict(os.environ, {"KAKAO_REST_API_KEY": "test-only"})
    @patch('api.pharmacies.httpx.AsyncClient.get', new_callable=AsyncMock)
    def test_upstream_errors(self, get):
        get.side_effect = httpx.ReadTimeout('timeout')
        self.assertEqual(client.get('/api/pharmacies?lat=37&lng=127').status_code, 504)
        get.side_effect = httpx.ConnectError('offline')
        self.assertEqual(client.get('/api/pharmacies?lat=37&lng=127').status_code, 502)
        for upstream, expected in [(401, 503), (403, 503), (429, 503), (500, 502)]:
            get.side_effect = None
            get.return_value = httpx.Response(upstream, request=httpx.Request('GET', 'https://dapi.kakao.com'))
            self.assertEqual(client.get('/api/pharmacies?lat=37&lng=127').status_code, expected)


if __name__ == '__main__':
    unittest.main()
