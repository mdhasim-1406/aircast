from fastapi.testclient import TestClient
from api.main import app


def test_health_is_typed_and_monitoring_honest():
    with TestClient(app) as c:
        assert c.get('/health').json() == {'status':'ok'}
        response = c.get('/monitoring')
        assert response.status_code == 200
        assert response.json()['status'] == 'insufficient_data'
        assert response.json()['performance']['status'] == 'insufficient_data'


def test_predict_requires_city_id():
    with TestClient(app) as c:
        assert c.post('/predict', json={}).status_code == 422
