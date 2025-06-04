import React from 'react';
import Image from 'next/image';
import { Roboto } from 'next/font/google'
import Link from 'next/link';

// Load Roboto with specific subsets and weights
const roboto = Roboto({
    subsets: ['latin'],
    weight: ['400', '700'], // Add other weights if needed
})

export default function Details({website}) {
    return (
        <div className="min-h-[81.1vh] w-full">
            <div className="h-auto w-full flex flex-col md:flex-row md:items-start gap-5 md:gap-10 px-5 sm:px-10 md:px-20 lg:px-30 py-5 md:py-0">
                <div className="w-full md:w-[40%]">
                    <Link href={`${website.url}`} target="_blank">
                        <div className="h-auto w-full shadow-xl rounded-[13px] overflow-hidden mt-5">
                            <Image src={website.image || '/default.jpeg'} alt={website.title || "Website title"} width={1440} height={1440} />
                        </div>
                    </Link>
                    <div className="mt-10 px-5 flex flex-col items-center gap-5">
                        <h1 className="text-lg">Overall Rating:</h1>
                        <div className="flex items-center justify-center gap-2">
                            {
                                Array.from({ length: website.rating }).map((_, i) => (
                                    <div key={i} className="h-10 w-10">
                                        <Image src={'/star.png'} alt="alt" width={1440} height={1440} />
                                    </div>
                                ))
                            }
                        </div>
                    </div>
                    <div className="mt-10 px-5 flex flex-col items-center gap-5">
                        <h1 className="text-lg">Trust Score:</h1>
                        <div className="w-full flex items-center justify-center gap-3">
                            <div className={`h-10 w-10 ${website.trust_score >= 80 && website.trust_score <= 100 && 'bg-green-500'} ${website.trust_score >= 50 && website.trust_score < 80 && 'bg-amber-400'} ${website.trust_score >= 0 && website.trust_score < 50 && 'bg-red-500'} rounded-[10px]`}></div>
                            <h1 className="text-3xl">{website.trust_score}</h1>
                        </div>
                    </div>
                    <div className="flex gap-4 items-center justify-center mt-10 text-sm">
                        <div className="flex items-center gap-2">
                            <div className="h-4 w-4 bg-green-500 rounded-[5px]"></div>
                            <h1 className="">Safe</h1>
                        </div>
                        <div className="flex items-center gap-2">
                            <div className="h-4 w-4 bg-amber-400 rounded-[5px]"></div>
                            <h1 className="">Caution</h1>
                        </div>
                        <div className="flex items-center gap-2">
                            <div className="h-4 w-4 bg-red-500 rounded-[5px]"></div>
                            <h1 className="">Unsafe</h1>
                        </div>
                    </div>
                </div>
                <div className="w-full md:w-[60%] py-5">
                    <div className="">
                        <h1 className="text-2xl font-extrabold"><span className="text-5xl"><span className="">#</span>{website.ranking}</span> {website.title}</h1>
                    </div>
                    <div className="mt-5">
                        <Link href={`${website.url}`} target="_blank">
                            <div className="px-5 py-2 bg-white dark:bg-black rounded-[12px] shadow-md md:shadow-lg cursor-pointer dark:border border-neutral-700">
                                <h1 className={`text-wrap text-sm ${roboto.className} tracking-wider text-center`}>{website.url}</h1>
                            </div>
                        </Link>
                        <div className="mt-5 flex gap-2">
                            {
                                (website.tags ?? []).map((tag, index) => {
                                    return (
                                        <div key={index} className="px-5 py-2 text-xs rounded-[10px] bg-neutral-200 dark:bg-neutral-700">{tag}</div>
                                    )
                                })
                            }
                        </div>
                        <div className="mt-10">
                            <h1 className="text-lg">Parametes</h1>
                            <div className="flex flex-col gap-2 mt-5 text-xl">
                                {
                                    (website.parameters ?? []).map((parameter, index) => {
                                        return (
                                            <div key={index} className="flex gap-5">
                                                <h1 className="font-bold">{parameter.name}:</h1>
                                                <h1 className="">{parameter.value}</h1>
                                            </div>
                                        )
                                    })
                                }
                            </div>
                        </div>
                        <div className="mt-10">
                            <h1 className="">Description</h1>
                            <div className="mt-5">
                                <p className="font-semibold">{website.description}</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
