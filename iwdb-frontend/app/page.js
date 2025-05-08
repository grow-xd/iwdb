'use client'
import React, { useEffect, useState } from 'react';
import Footer from '@/components/LandingPageComponents/Footer';
import Menu from '@/components/LandingPageComponents/Menu';
import SearchBar from '@/components/LandingPageComponents/SearchBar';
import WebsitesList from '@/components/LandingPageComponents/WebsitesList';
// import data from '@/components/LandingPageComponents/WebsiteData';
import { RiArrowRightLine, RiCloseLine, RiMenuLine, RiSearchLine } from '@remixicon/react';
import Link from 'next/link';

export default function Home() {
  const [search, setSearch] = useState('');
  const [isValidLink, setIsValidLink] = useState(false);
  const [link, setLink] = useState('');
  const [menu, setMenu] = useState(false);
  const [data, setData] = useState([]);
  const [message, setMessage] = useState('');

  useEffect(() => {
    const callAPI = async () => {
      const response = await fetch('http://127.0.0.1:8000/api/items/');
      if(response.ok) {
        const json_data = await response.json();
        setData(json_data);
        setMessage('successfully fetched data');
      } else {
        setMessage('error fetching data');
      }
      console.log(message);
    }
    callAPI();
  }, [message]);

  return (
    <div className="relative min-h-screen w-full bg-neutral-100 dark:bg-neutral-900 text-neutral-900 dark:text-neutral-100 transition delay-300 ease-in-out">
      <div className="fixed md:hidden h-[7vh] w-auto z-40 px-2 flex items-center justify-center">
        {!menu ? (<div onClick={() => setMenu(true)} className="p-2 rounded-[10px] bg-neutral-200/50">
          <RiMenuLine height={20} width={20} />
        </div>):(
        <div onClick={() => setMenu(false)} className="p-2 rounded-[10px] bg-neutral-200/50">
          <RiCloseLine height={20} width={20} />
        </div>
      )}
      </div>
      <Menu />
      <div className="h-[7vh] w-full"></div>
      <SearchBar search={search} setSearch={setSearch} isValidLink={isValidLink} setIsValidLink={setIsValidLink} link={link} setLink={setLink} />
      <div className="h-[10vh] w-full"></div>
      {menu && (<div className="fixed md:hidden h-auto w-full px-2 py-2 -mt-5">
        <div className="h-auto w-full rounded-[10px] bg-neutral-100 py-5 px-5 flex flex-col gap-3">
          <Link href="/about"><div className="px-5 py-2 border border-neutral-400 rounded-[10px] text-center">About</div></Link>
          <Link href="/guide"><div className="px-5 py-2 border border-neutral-400 rounded-[10px] text-center">Guide</div></Link>
          <Link href="/support"><div className="px-5 py-2 border border-neutral-400 rounded-[10px] text-center">Support</div></Link>
        </div>
      </div>)}
      <div className="min-h-[76vh] w-full">
        {data.length > 0 && data.sort((a,b) => a.ranking - b.ranking).filter(website => website.url.includes(search) || website.title.includes(search)).length > 0 && <WebsitesList search={search} data={data} />}
        {data.length > 0 && data.sort((a,b) => a.ranking - b.ranking).filter(website => website.url.includes(search) || website.title.includes(search)).length === 0 && (
          <div className="w-full h-[76vh] flex items-center justify-center px-5">
            <div className="flex flex-col items-center gap-3 select-none">
              {isValidLink ? (
                <>
                  <h1 className="text-center text-2xl font-semibold text-neutral-400 dark:text-neutral-600">{link} has not been processed yet.</h1>
                  <h2 className="text-center text-4xl font-semibold text-neutral-400 dark:text-neutral-600">Please proceed to process this link</h2>
                </>
              ):(
                <>
                  <h1 className="text-center text-2xl font-semibold text-neutral-400 dark:text-neutral-600">Link not found</h1>
                  <h2 className="text-center text-4xl font-semibold text-neutral-400 dark:text-neutral-600">Please enter a valid link to proceed</h2>
                </>
              )}
              {!isValidLink && (<div className="flex items-center gap-10 mt-5">
                <div className="p-3 bg-neutral-200 dark:bg-neutral-700 rounded-[10px]">
                  <RiSearchLine />
                </div>
                <div className="text-neutral-400 dark:text-neutral-600">
                  <RiArrowRightLine height={20} width={20} />
                </div>
                <div className="p-3 bg-neutral-200 dark:bg-neutral-700 rounded-[10px]">
                  <RiArrowRightLine />
                </div>
              </div>)}
            </div>
          </div>
        )}
        {data.length == 0 && (
          <div className="w-full h-[76vh] flex items-center justify-center px-5">
            <div className="flex flex-col items-center gap-3 select-none">
              {isValidLink ? (<>
                  <h1 className="text-center text-2xl font-semibold text-neutral-400 dark:text-neutral-600">This link has not been processed yet.</h1>
                  <h2 className="text-center text-4xl font-semibold text-neutral-400 dark:text-neutral-600">Please proceed to process this link</h2>
                </>):(
                  <>
                    <h1 className="text-center text-2xl font-semibold text-neutral-400 dark:text-neutral-600">No links available</h1>
                    <h2 className="text-center text-4xl font-semibold text-neutral-400 dark:text-neutral-600">Please enter a valid link and proceed</h2>
                  </>
                )
              }
              {!isValidLink && (<div className="flex items-center gap-10 mt-5">
                <div className="p-3 bg-neutral-200 dark:bg-neutral-700 rounded-[10px]">
                  <RiSearchLine />
                </div>
                <div className="text-neutral-400 dark:text-neutral-600">
                  <RiArrowRightLine height={20} width={20} />
                </div>
                <div className="p-3 bg-neutral-200 dark:bg-neutral-700 rounded-[10px]">
                  <RiArrowRightLine />
                </div>
              </div>)}
            </div>
          </div>
        )

        }
      </div>
      <Footer />
    </div>
  );
}
