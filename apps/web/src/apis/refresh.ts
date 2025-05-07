import { AxiosError } from 'axios';

const refresh = async (error: AxiosError) => {
  if (error.response?.status === 401) {
    localStorage.setItem('SPEC_ACCESS_TOKEN', '');
    localStorage.setItem('SPEC_USER_STORE', '');
    window.location.href = '/login';
  }

  return Promise.reject(error);
};

export default refresh;
