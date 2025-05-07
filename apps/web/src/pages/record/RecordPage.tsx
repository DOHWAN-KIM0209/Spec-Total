import Button from '@/components/@common/Button/Button';
import InterviewQuestion from '@/components/record/InterviewQuestion';
import RecordTimer from '@/components/record/RecordTimer';
import RecordSetting from '@/components/record/RecordSetting';
import { useInterview } from '@/hooks/interview/useInterview';
import { ChangeEvent, useCallback, useEffect, useRef, useState } from 'react';
import BackgroundOpacity from '@/components/record/BackgroundOpacity';
import RecordUploading from '@/components/record/RecordUploading';
import { useSpeechRecognition } from 'react-speech-kit';
import { GrNotes } from 'react-icons/gr';
import CheatSheetModal from '@/components/record/CheatSheetModal';
import { IInterviewAnalyzeReq, IInterviewFollowupReq, IInterviewQuestionItem } from '@/types/interview';
import { convertBlobToFile } from '@/utils/convertBlobToFile';
import { getCurrentTime } from '@/utils/getCurrentTime';
import Lottie from 'react-lottie';
import followupLoading from '@/assets/lotties/followupLoading.json';
import { useLocation, useNavigate } from 'react-router-dom';
import questionStore from '@/stores/questionStore';

export type recordStatusType = 'pending' | 'preparing' | 'recording' | 'proceeding' | 'uploading' | 'ending';

const RecordPage = () => {
  const { useGetMainInterviewQuestionList, usePostInterviewAnalyze, usePostFollowupQuestion, usePostInterviewSet } =
    useInterview();
  const { data, mutate: getQuestionList } = useGetMainInterviewQuestionList();
  const { mutate: postInterviewAnalyze } = usePostInterviewAnalyze();
  const { data: followupQuestion, mutate: postFollowupQuestion } = usePostFollowupQuestion();
  const { data: interviewSet, mutate: postInterviewSet } = usePostInterviewSet();
  const { resetQuestion } = questionStore();
  const [questionList, setQuestionList] = useState<IInterviewQuestionItem[]>([]);
  const [questionIndex, setQuestionIndex] = useState<number>(0); // 질문 인덱스 상태 관리
  const { state } = useLocation();
  const navigate = useNavigate();

  useEffect(() => {
    if (data) {
      setQuestionList(data.data.questionList);
    }
  }, [data]);
  /*
    stream 연결 전 상태는 stream으로 판단
    pending -> 카메라, 마이크 세팅 중 (사용자가 파란색 다음버튼을 눌렀을 때 다음 상태로 변경)
    preparing -> 질문을 받아와 녹화대기 중인 상태 (사용자가 녹화시작 버튼을 눌렀을 때 다음 상태로 변경)
    recording -> 이후 RecordTimer 컴포넌트에서 카운트 다운 시작하고 종료하면 녹화 시작
    proceeding -> 녹화 진행중인 상태
    uploading -> 녹화한 영상을 서버에 업로드하는 중 (사용자가 직접 다음을 누르거나 타이머가 다 지났을 때 다시 recording 상태로 변경)
    ending -> 전체 질문을 모두 답변 한 후 상태
  */
  const [status, setStatus] = useState<recordStatusType>('pending');

  // "거울",  화면에 보여지는 역할
  const videoRef = useRef<HTMLVideoElement>(null);

  const [stream, setStream] = useState<MediaStream | null>(null);

  useEffect(() => {
    if (state) {
      postInterviewSet({ type: state.type, startTime: state.startTime });

      if (state.questionList) {
        setQuestionList(state.questionList);
      } else {
        getQuestionList(state.resumeId);
      }
    }
  }, [state]);

  const getMediaPermission = useCallback(async () => {
    try {
      const video = { audio: true, video: true };

      const videoStream = await navigator.mediaDevices.getUserMedia(video);
      setStream(videoStream);

      if (videoRef.current) {
        videoRef.current.srcObject = videoStream;
      }
    } catch (err) {
      console.log(err);
    }
  }, []);

  // 버튼 상태 관리
  const [btnText, setBtnText] = useState<string>('다음');
  const getButtonText = useCallback(() => {
    switch (status) {
      case 'preparing':
        setBtnText('녹화시작');
        break;
      case 'uploading':
        setBtnText('');
        break;
      case 'proceeding':
        setBtnText('녹화종료');
        break;
      case 'recording':
        setBtnText('');
        break;
    }
  }, [status]);

  const nextButtonClick = async () => {
    if (!stream) {
      alert('카메라와 마이크를 확인해보세요');
      return;
    }

    if (status === 'pending') {
      setStatus('preparing');
    } else if (status === 'preparing') {
      setStatus('recording');
      setStt('');
    } else if (status === 'proceeding') {
      setStatus('uploading');
      handleStopRecording();
    }
  };

  useEffect(() => {
    getButtonText();

    switch (status) {
      case 'preparing':
        break;
      case 'uploading':
        setIsOpen(false);
        break;
      case 'proceeding': // 녹화 및 STT 시작
        handleStartRecording();
        listen({ interimResults: false, lang: 'ko-KR' });
        break;
      case 'recording':
        setIsOpen(false);
        break;
    }
  }, [status]);

  // dom내부에선 보여지지 않지만, 내부에서 "녹화"에 대한 기능만을 수행합니다.
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const [recordedBlobs, setRecordedBlobs] = useState<Blob[]>([]);

  const handleStartRecording = () => {
    setRecordedBlobs([]);

    try {
      // MediaRecorder 생성자 호출
      mediaRecorderRef.current = new MediaRecorder(stream as MediaStream, {
        mimeType: 'video/webm; codecs=vp9',
      });

      // 전달받은 데이터를 처리
      // 녹화된 미디어 데이터가 사용 가능할 때 트리거됩니다. (handleStopRecording 호출 시)
      mediaRecorderRef.current.ondataavailable = event => {
        if (event.data && event.data.size > 0) {
          setRecordedBlobs(prev => [...prev, event.data]);
        }
      };
      mediaRecorderRef.current.start();
    } catch (e) {
      console.log('MediaRecorder error');
    }
  };

  // 녹화 종료 시
  const handleStopRecording = () => {
    if (mediaRecorderRef.current) {
      // stt 중지
      stop();

      // (실전면접 + 이력서 기반 질문)이라면 꼬리 질문 생성
      if (state.type === 'main' && questionList[questionIndex].type === 'resume') {
        const req: IInterviewFollowupReq = {
          answer: stt,
          question: questionList[questionIndex].question,
        };
        setIsFollowup(true);
        postFollowupQuestion(req);
      }

      // 녹화 중지
      mediaRecorderRef.current.stop();
    }
  };

  // 꼬리질문 생성중인지 상태 관리
  const [isFollowup, setIsFollowup] = useState<boolean>(false);
  // 꼬리질문 생성 후 기존 list에 추가하기
  useEffect(() => {
    if (followupQuestion) {
      const newData = [...questionList]; // 기존 데이터를 복사합니다.

      const req: IInterviewQuestionItem = {
        question: followupQuestion.data.followUpQuestion.question,
        type: 'followup',
        keywordList: [],
      };
      newData.splice(questionIndex, 0, req);

      setQuestionList(newData);
      setIsFollowup(false);
    }
  }, [followupQuestion]);

  // 녹화 중지 후 분석 요청
  useEffect(() => {
    if (recordedBlobs.length) {
      const check = questionList[questionIndex];
  
      // 💥 이 한 줄이 핵심: undefined면 더 이상 진행하지 않게 막는다
      if (!check || !interviewSet?.data?.interview) return;
  
      const time = getCurrentTime();
      const filename = questionIndex + '_' + time + '.mp4';
      const videoFile = convertBlobToFile(recordedBlobs, filename);
  
      const req: IInterviewAnalyzeReq = {
        interviewId: interviewSet.data.interview.id,
        question: check.question,
        questionType: check.type,
        answer: stt,
        skip: skip,
        keywordList: check.keywordList.map(item => item.keyword),
      };
  
      const formData = new FormData();
      const json = JSON.stringify(req);
  
      formData.append('analysisRequestDto', json);
      formData.append('video', videoFile);
      postInterviewAnalyze(formData);
    }
  }, [recordedBlobs]);
  

  useEffect(() => {
    // 아직 media stream이 설정되지 않았다면 호출
    if (!stream) {
      getMediaPermission();
    }

    // 페이지 이탈 또는 컴포넌트 언마운트 시 미디어 스트림 정지
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [stream]);

  // STT 상태 관리
  const [stt, setStt] = useState<string>('');
  const { listen, stop } = useSpeechRecognition({
    onResult: (result: string) => {
      // 음성인식 결과가 value 상태값으로 할당
      setStt(prev => {
        return prev + ' ' + result;
      });
    },
  });

  // 모달 상태 관리
  const [isOpen, setIsOpen] = useState<boolean>(false);

  // 타이머 세팅 상태 관리
  const [timerSetting, setTimerSetting] = useState<number>(30);
  const radioHandler = (e: ChangeEvent<HTMLInputElement>) => {
    setTimerSetting(Number(e.target.value));
  };

  // skip 상태 관리 (10초 이내면 true)
  const [skip, setSkip] = useState<boolean>(false);
  const followupOptions = {
    loop: true,
    autoplay: true,
    animationData: followupLoading, // Lottie 애니메이션 데이터
    rendererSettings: {
      preserveAspectRatio: 'xMidYMid slice',
    },
  };

  return (
    <div>
      <div className="flex flex-col justify-center items-center min-w-[58rem] h-screen pb-10">
        <div className="w-[58rem] h-14 relative flex justify-end pb-5 text-center">
          <p className="absolute top-0 left-1/2 -translate-x-1/2 font-bold text-3xl">
            {!stream ? '카메라, 마이크를 준비중입니다' : status === 'pending' ? '다음 버튼을 눌러주세요' : ''}
          </p>
          {btnText && !isFollowup && status !== 'ending' && (
            <Button
              text={btnText}
              height="h-9"
              backgroundColor={stream ? 'bg-MAIN1' : 'bg-gray-200'}
              textColor="text-white"
              onClick={() => nextButtonClick()}
            />
          )}
        </div>
        <div className="relative bg-black px-[9rem]">
          <video className="w-[40rem] h-[30rem] -scale-x-100" ref={videoRef} autoPlay />

          {stream && status === 'pending' && <RecordSetting />}

          {stream && status === 'preparing' && !isFollowup && (
            <>
              <BackgroundOpacity />
              {questionList && (
                <InterviewQuestion
                  timerSetting={timerSetting}
                  question={questionList[questionIndex]}
                  setStatus={setStatus}
                  setSkip={setSkip}
                  status={status}
                  handleStopRecording={handleStopRecording}
                />
              )}
              <div className="absolute bottom-0 w-[40rem]">
                <div className="flex justify-around text-white m-6 p-5  text-center bg-black/50 rounded-lg">
                  <p>녹화 시간 설정</p>
                  <div className="flex gap-5">
                    <div className="flex gap-2">
                      <input id="thirty" type="radio" value={30} onChange={radioHandler} checked={timerSetting == 30} />
                      <label htmlFor="thirty">30초</label>
                    </div>
                    <div className="flex gap-2">
                      <input id="sixty" type="radio" value={60} onChange={radioHandler} checked={timerSetting == 60} />
                      <label htmlFor="sixty">60초</label>
                    </div>
                    <div className="flex gap-2">
                      <input id="ninety" type="radio" value={90} onChange={radioHandler} checked={timerSetting == 90} />
                      <label htmlFor="ninety">90초</label>
                    </div>
                  </div>
                </div>
              </div>
              {state.type === 'mock' && (
                <GrNotes
                  onClick={() => setIsOpen(!isOpen)}
                  size={20}
                  color="white"
                  className="absolute bottom-0 right-0 m-6 cursor-pointer"
                />
              )}
              {isOpen && questionList && (
                <CheatSheetModal question={questionList[questionIndex]} setIsOpen={setIsOpen} />
              )}
            </>
          )}

          {stream && status === 'recording' && (
            <>
              <BackgroundOpacity />
              <RecordTimer setStatus={setStatus} />
            </>
          )}

          {stream && questionList && status === 'proceeding' && (
            <>
              <InterviewQuestion
                timerSetting={timerSetting}
                question={questionList[questionIndex]}
                setStatus={setStatus}
                setSkip={setSkip}
                status={status}
                handleStopRecording={handleStopRecording}
              />
              {state.type === 'mock' && (
                <GrNotes
                  onClick={() => setIsOpen(!isOpen)}
                  size={20}
                  color="white"
                  className="absolute bottom-0 right-0 m-6 cursor-pointer"
                />
              )}
              {isOpen && questionList && (
                <CheatSheetModal question={questionList[questionIndex]} setIsOpen={setIsOpen} />
              )}
            </>
          )}

          {stream && questionList && status === 'uploading' && (
            <>
              <RecordUploading
                questionList={questionList}
                questionIndex={questionIndex}
                setQuestionIndex={setQuestionIndex}
                setStatus={setStatus}
              />
            </>
          )}

          {isFollowup && (
            <>
              <div className="absolute top-0 left-0 right-0 bottom-0 bg-black z-50">
                <div className="w-full h-full flex flex-col items-center justify-center gap-10">
                  <p className="text-center text-3xl text-white">답변을 업로드하고 있습니다</p>
                  <Lottie options={followupOptions} height={150} width={150} />
                </div>
              </div>
            </>
          )}

          {status === 'ending' && (
            <>
              <BackgroundOpacity />
              <div className="absolute top-0 left-0 right-0 bottom-0 ">
                <div className="w-full h-full flex flex-col text-center items-center justify-center gap-5">
                  <p className="text-3xl text-white">면접이 종료되었습니다</p>
                  <p className="text-3xl text-white pb-5">수고하셨습니다</p>
                  <Button
                    width="w-24"
                    text="결과 확인"
                    height="h-10"
                    backgroundColor="bg-gray-50"
                    textColor="text-black"
                    textSize="text-xs"
                    onClick={() => {
                      resetQuestion();
                      navigate('/result');
                    }}
                  />
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default RecordPage;
