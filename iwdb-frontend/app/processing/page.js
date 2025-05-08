'use client'
import React,{useEffect, useState, use} from 'react';
import { useRouter } from 'next/navigation';

export default function ProcessingPage() {
    const [link, setLink] = useState(null);
    const router = useRouter(null);
    useEffect(() => {
        if(link === null && sessionStorage.getItem('link') !== null) {
            setLink(sessionStorage.getItem('link'));
        }
    },[router, link]);

    const handleClick = () => {
        sessionStorage.removeItem('link');
        router.replace('/');
    }

    return (
        <div className="min-h-screen w-full bg-neutral-100 dark:bg-neutral-900 text-neutral-900 dark:text-neutral-100 transition delay-300 ease-in-out">
            <div className="h-screen w-full flex flex-col items-center justify-center">
                Processing Page....
                <div className="font-light text-neutral-400">{link}</div>
                <button onClick={handleClick} className="mt-5 px-5 py-2 rounded-[10px] text-sm bg-blue-500 text-white cursor-pointer">Home</button>
            </div>
        </div>
    )
}
