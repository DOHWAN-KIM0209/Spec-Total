import { RouterProvider, createBrowserRouter } from 'react-router-dom';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import HomePage from './pages/home/HomePage';
import InterviewPage from './pages/interview/InterviewPage';
import LoginPage from './pages/login/LoginPage';
import MyPage from './pages/my/MyPage';
import PrivateRoute from './pages/private/PrivateRoute';
import QuestionPage from './pages/question/QuestionPage';
import RecordPage from './pages/record/RecordPage';
import ResultPage from './pages/result/ResultPage';
import ResultReportPage from './pages/result/ResultReportPage';
import SignUpPage from './pages/signup/SignupPage';
import ResumeSelectPage from './pages/interview/ResumeSelectPage';
import ErrorPage from './pages/error/ErrorPage';
import { useEffect } from 'react';
import Toast from './components/@common/Toast/Toast';
import { EventSourcePolyfill } from 'event-source-polyfill';
import userStore from './stores/userStore';
import { useQueryClient } from '@tanstack/react-query';
import SecondSection from './components/home/SecondSection';

const router = createBrowserRouter([
  {
    path: '/',
    element: <HomePage />,
  },
  {
    path: '/signup',
    element: <SignUpPage />,
  },
  {
    path: '/login',
    element: <LoginPage />,
  },
  {
    path: '/second',
    element: <SecondSection />,
  },
  {
    path: '/my',
    element: (
      <PrivateRoute>
        <MyPage />
      </PrivateRoute>
    ),
  },
  {
    path: '/interview',
    element: <InterviewPage />,
  },
  {
    path: '/interview/resume',
    element: (
        <ResumeSelectPage />
    ),
  },
  {
    path: '/record',
    element: (
        <RecordPage />
    ),
  },
  {
    path: '/question',
    element: <QuestionPage />,
  },
  {
    path: '/question-list',
    element: <QuestionPage />,
  },
  {
    path: '/result',
    element: (
        <ResultPage />
    ),
  },
  {
    path: '/result/:id',
    element: (
        <ResultReportPage />
    ),
  },
  {
    path: '/*',
    element: <ErrorPage />,
  },
]);

function App() {
  const { isLogin } = userStore();
  const queryClient = useQueryClient();
  useEffect(() => {
    if (isLogin) {
      // eventSource 객체 생성
      const eventSource = new EventSourcePolyfill(import.meta.env.VITE_BASE_URL + '/file/sse/v1', {
        headers: {
          Authorization: 'Bearer ' + localStorage.getItem('SPEC_ACCESS_TOKEN'),
          Connetction: 'keep-alive',
          'Content-Type': 'text/event-stream',
        },
        heartbeatTimeout: 86400000,
      });

      // eventSource Connection 됐을때
      eventSource.onopen = () => {};

      // eventSource 에러 시 할 일
      eventSource.onerror = async event => {
        console.log(event);
        eventSource.close();
      };

      eventSource.addEventListener('spec', async function (event: any) {
        const data = JSON.parse(event.data);

        if (data && data.data.resume) {
          queryClient.invalidateQueries({ queryKey: ['ResumeList'] });
          queryClient.invalidateQueries({ queryKey: ['questionsInfinite'] });
          Toast.success(`'${data.data.resume}'` + ' 질문이 생성되었습니다.');
        }
      });

      eventSource.onmessage = async event => {
        console.log(event);
      };

      return () => {
        eventSource.close();
      };
    }
  }, [isLogin]);

  return (
    <>
      <ToastContainer />
      <RouterProvider router={router} />
    </>
  );
}

export default App;
