import React, { useState, useEffect } from 'react';

/** Reveals `text` one character at a time. Safe with null/undefined input. */
const Typewriter = ({ text, speed = 10 }) => {
  const safeText = typeof text === 'string' ? text : '';
  const [displayedText, setDisplayedText] = useState('');

  useEffect(() => {
    setDisplayedText('');
    if (!safeText) return undefined;

    let i = 0;
    const timer = setInterval(() => {
      i += 1;
      setDisplayedText(safeText.substring(0, i));
      if (i >= safeText.length) clearInterval(timer);
    }, speed);

    return () => clearInterval(timer);
  }, [safeText, speed]);

  return <span>{displayedText}</span>;
};

export default Typewriter;
