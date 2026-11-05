export function ThemeInit() {
  const script = `(function(){try{var t=localStorage.getItem("fplpredict-theme");if(t==="light")document.documentElement.dataset.theme="light";}catch(e){}})();`;
  return <script dangerouslySetInnerHTML={{ __html: script }} />;
}
