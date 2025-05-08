import axios from 'axios';
import refresh from './refresh';
import setAuthorization from './setAuthorization';

const baseURL = import.meta.env.VITE_BASE_URL; // ✅ 조건문 제거

const axiosRequestConfig = { baseURL };
const axiosWithCredentialConfig = { baseURL, withCredentials: true };
const axiosFileConfig = {
  baseURL,
  headers: { 'Content-Type': 'multipart/form-data' },
  withCredentials: true,
};

// 인스턴스 정의
export const axiosCommonInstance = axios.create(axiosRequestConfig);
export const axiosWithCredentialInstance = axios.create(axiosWithCredentialConfig);
export const axiosAuthInstance = axios.create(axiosWithCredentialConfig);
export const axiosFileInstance = axios.create(axiosFileConfig);

// 인터셉터 연결
[axiosAuthInstance, axiosWithCredentialInstance, axiosFileInstance].forEach(instance => {
  instance.interceptors.request.use(setAuthorization);
  instance.interceptors.response.use(undefined, refresh);
});
