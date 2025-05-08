'use client'
import React, { use, useState, useEffect } from 'react';
import Menu from '@/components/LandingPageComponents/Menu';
import Footer from '@/components/LandingPageComponents/Footer';
import Details from '@/components/DetailsPageComponents/Details';
import { RiArrowLeftLine, RiCloseLine, RiMenuLine } from '@remixicon/react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
// import data from '@/components/LandingPageComponents/WebsiteData';


export default function DetailsPage({params}) {
    const router = useRouter();
    const p = use(params);
    const [menu, setMenu] = useState(false);

    const [website, setWebsite] = useState([]);
    const [message, setMessage] = useState('');

    useEffect(() => {
        const callAPI = async () => {
            const response = await fetch(`http://127.0.0.1:8000/api/items/${p.id}/`);
            if(response.ok) {
                const json_website = await response.json();
                setWebsite(json_website);
                setMessage('successfully fetched data');
            } else {
                setMessage('error fetching data');
            }
            console.log(message);
        }
        callAPI();
    }, [p, message]);

    // const website = data.find((site) => site.id.toString() === p.id);
    const handleReturnClick = () => {
        router.replace('/');
    }

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
            {menu && (<div className="fixed md:hidden h-auto w-full px-2 py-2 -mt-5 z-10">
                <div className="h-auto w-full rounded-[10px] bg-neutral-100 py-5 px-5 flex flex-col gap-3">
                    <Link href="/about"><div className="px-5 py-2 border border-neutral-400 rounded-[10px] text-center">About</div></Link>
                    <Link href="/guide"><div className="px-5 py-2 border border-neutral-400 rounded-[10px] text-center">Guide</div></Link>
                    <Link href="/support"><div className="px-5 py-2 border border-neutral-400 rounded-[10px] text-center">Support</div></Link>
                </div>
            </div>)}
            <div onClick={handleReturnClick} className="fixed mt-2 ml-5 h-10 w-10 bg-neutral-200 dark:bg-neutral-800 rounded-[13px] flex items-center justify-center cursor-pointer ">
                <RiArrowLeftLine height={20} width={20}/>
            </div>
            <div className="h-10"></div>
            <Details website={website} />
            <Footer />
        </div>
    )
}
