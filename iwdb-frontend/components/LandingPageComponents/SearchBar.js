'use client'
import React from 'react';
import {
    RiArrowRightLine,
    RiSearchLine
} from '@remixicon/react';
import { useRouter } from 'next/navigation';

export default function SearchBar({isValidLink, search, setSearch, setIsValidLink, setLink}) {
    const router = useRouter();
    const handleClick = () => {
        router.push(`/processing`);
    }

    const handleSearchChange = (e) => {
        setSearch(e.target.value);
        let regex = /^(https?:\/\/)?([\da-z.-]+)\.([a-z.]{2,6})([\/\w .-]*)*\/?$/;
        setIsValidLink(regex.test(e.target.value));
        if(isValidLink) {
            setLink(e.target.value);
            sessionStorage.setItem('link', e.target.value);
        } else {
            sessionStorage.removeItem('link');
        }
    }

    return (
        <div className="fixed h-[10vh] w-full flex items-center justify-center bg-neutral-100 dark:bg-neutral-900 z-10">
            <div className="h-10 w-[90%] sm:w-[70%] md:w-[60%] lg:w-[50%] border-[0.5px] border-neutral-800 dark:border-neutral-500 rounded-[10px] flex overflow-hidden">
                <div className="w-[90%] h-full flex items-center">
                    <input
                        type="text"
                        placeholder="Enter your links here...."
                        value={search}
                        onChange={handleSearchChange}
                        className="h-[50%] w-full px-5 flex resize-none outline-none"
                    />
                </div>
                {isValidLink ? (
                    <div onClick={handleClick} className="w-[10%] h-full bg-neutral-200 dark:bg-neutral-700 flex items-center justify-center cursor-pointer">
                        <RiArrowRightLine height={20} width={20} />
                    </div>
                ): (
                    <div className="w-[10%] h-full bg-neutral-200 dark:bg-neutral-700 flex items-center justify-center cursor-pointer">
                        <RiSearchLine height={20} width={20} />
                    </div>
                )}
            </div>
        </div>
    );
}
