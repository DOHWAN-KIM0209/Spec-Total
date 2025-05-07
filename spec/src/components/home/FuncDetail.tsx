import DetailBox from './DetailBox';
import resume from '@/assets/images/resume.gif';
import result from '@/assets/images/result.gif';
import mock from '@/assets/images/mock.gif';
import main from '@/assets/images/main.gif';

interface IFunction {
  type: 'left' | 'right';
  subTitle?: React.ReactNode;
  title?: React.ReactNode;
  description?: React.ReactNode;
  buttonText: string;
  navigateUrl: string;
  video: any;
}

const FuncDetail = () => {
  const funcInfo: IFunction[] = [
    {
      type: 'left',
      subTitle: '이력서 기반 질문 생성',
      title: (
        <>
          AI가 추천하는
          <br />
          나에게 꼭 필요한 면접 질문 생성
        </>
      ),
      description: (
        <>
          내 이력서를 업로드하면,
        <br />이력서를 기반으로 면접 질문이 자동으로 생성됩니다.
        <br />원하는 질문을 리스트에 담아 면접 연습에 활용해요.
        </>
      ),
      buttonText: '이력서 등록하기',
      navigateUrl: '/my',
      video: <img src={resume} className="border" />,
    },
    {
      type: 'right',
      subTitle: '면접 연습 및 녹화',
      title: (
        <>
          나만의 질문 리스트로
          <br />
          진행하는 면접 연습 및 녹화
        </>
      ),
      description: (
        <>
          면접 연습 모드에서는 3개의 질문을 선택하여 실력을 확인할 수 있습니다.
        <br />실전 모드에서는 질문 선택 없이, 공통 질문과 이력서 기반 질문으로 실제 면접처럼 진행됩니다.
        <br />녹화된 영상은 분석 리포트 작성에 활용됩니다.
        </>
      ),
      buttonText: '면접 연습하러가기',
      navigateUrl: '/interview',
      video: <img src={mock} className="border" />,
    },
    {
      type: 'left',
      subTitle: '무작위 꼬리 질문',
      title: <>내 답변 기반 실시간 꼬리 질문</>,
      description: (
        <>
          답변에 따라 실시간으로 변화하는 꼬리 질문에 대비하세요.
          <br />AI가 사전 질문이 아닌, 당신의 답변을 분석해 맞춤형 추가 질문을 생성합니다.
        </>
      ),
      buttonText: '실전 면접 연습하러가기',
      navigateUrl: '/interview',
      video: <img src={main} className="border" />,
    },
    {
      type: 'right',
      subTitle: '분석 보고서',
      title: (
        <>
          내 면접 영상을 녹화하고
          <br />
          영상 분석을 통한 피드백까지
        </>
      ),
      description: (
        <>
          면접이 끝난 후, 영상 기반 분석을 통해 합격 가능성과 세부 결과를 확인할 수 있어요.
          <br />질문별로 감정, 의도, 키워드 포함 여부까지 분석해서 더 나은 답변을 준비할 수 있어요.
        </>
      ),
      buttonText: '기존 영상 분석 결과 확인하기',
      navigateUrl: '/result',
      video: <img src={result} className="border" />,
    },
  ];

  return (
    <div className="w-full">
      {funcInfo.map((ele, idx) => {
        return <DetailBox key={idx} info={ele} />;
      })}
    </div>
  );
};

export default FuncDetail;
