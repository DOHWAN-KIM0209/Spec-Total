import logo from '@/assets/images/logo.png';
import main from '@/assets/images/main.webp';

import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import Button from '../@common/Button/Button';

const FirstSection = () => {
  const navigate = useNavigate();


  return (
    <>
      <img src={main} alt="main" className="w-screen h-screen absolute left-0 top-0 z-[-1] object-cover" />
      <div className="w-screen h-screen absolute left-0 top-0 bottom-0 z-[-1] bg-white/10" />
      <motion.div
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        viewport={{ once: true }}
        transition={{ duration: 1.5 }}
      >
        <div className="w-screen h-screen absolute left-0 top-0 bottom-0 z-[-1] bg-gradient-to-r from-white/70 to-white/20"></div>
        <div className="w-screen h-screen text-[25px] leading-14 font-extrabold text-BLACK pl-40 flex flex-col justify-center z-10">
          <img src={logo} alt="logo" width={'270px'} />
          <div className="text-[18px] py-4">
            모의 면접 분석기로 면접 질문에 답하며 직접 연습해보세요!
            <br /> 표정 분석까지 제공되어 더 자신감 있게 면접에 대비할 수 있어요! <br />
          </div>
          <Button
            text="시작하기"
            width="w-[180px]"
            height="h-[50px]"
            backgroundColor="bg-[#01A255]"
            hoverBackgroundColor="hover:bg-[#018B4A]"
            textColor="text-white"
            textSize="text-[18px]"
            style={{ fontWeight: '300' }}
            onClick={() => navigate('/second')}
          />
        </div>
      </motion.div>
    </>
  );
};

export default FirstSection;
